from torch_scope import wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import argparse
import logging
import functools
import json

from betanlp.model import seqLabel, seqLabelEvaluator
from betanlp.encoder import strFromFileEncoderWrapper
from betanlp.common.utils import adjust_learning_rate
from betanlp.optim import Nadam

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/twitter_ner.json")
    conf = parser.parse_args()
    with open(conf.config, 'r') as fin:
        args = json.load(fin)

    pw = wrapper(os.path.join(args["cp_root"], args["checkpoint_name"]), args["checkpoint_name"], enable_git_track=args["git_tracking"])

    gpu_index = pw.auto_device() if 'auto' == args["gpu"] else int(args["gpu"])
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)

    logger.info("Exp: {}".format(args['checkpoint_name']))
    logger.info("Config: {}".format(args))

    logger.info('Saving the configure...')
    pw.save_configue(args)

    logger.info('Building the model...')
    model = seqLabel(args)
    evaluator = seqLabelEvaluator(model.spDecoder.to_spans)

    logger.info('Loading the data...')
    train_data = strFromFileEncoderWrapper(args, processed_file = args['train_file'])
    dev_data = strFromFileEncoderWrapper(args, processed_file = args['dev_file'])

    if args['test_file']:
        test_data = strFromFileEncoderWrapper(args, processed_file = args['test_file'])
    else:
        test_data = None

    logger.info('Loading to GPU: {}'.format(gpu_index))
    model.to(device)

    print(model)
    # set_trace()

    logger.info('Constructing optimizer')
    param_dict = filter(lambda t: t.requires_grad, model.parameters())
    # optim_map = {'Nadam': Nadam, 'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': functools.partial(optim.SGD, momentum=0.9), 'RMSprop': optim.RMSprop}
    optim_map = {'Nadam': Nadam, 'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta, 'SGD': optim.SGD, 'RMSprop': optim.RMSprop}
    weight_decay = 0.0 if 'weight_decay' not in args else args['weight_decay']
    if args['lr'] > 0:
        optimizer=optim_map[args['update']](param_dict, lr=args['lr'], weight_decay=weight_decay)
    else:
        optimizer=optim_map[args['update']](param_dict, weight_decay=weight_decay)

    logger.info('Setting up training environ.')
    best_f1 = float('-inf')
    patience_count = 0
    batch_index = 0
    normalizer=0
    tot_loss = 0

    for indexs in range(args['epoch']):

        logger.info('###### {} ######'.format(args['checkpoint_name']))
        logger.info('Epoch: {}'.format(indexs))
        # pw.nvidia_memory_map()

        model.train()
        for x, y in train_data.get_tqdm(device, args['batch_size']):

            model.zero_grad()
            loss = model(x, y)['loss']

            tot_loss += loss.item()
            normalizer += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
            optimizer.step()

            batch_index += 1
            if 0 == batch_index % 100:
                pw.add_loss_vs_batch({'training_loss': tot_loss / (normalizer + 1e-9)}, batch_index, use_logger = False)
                tot_loss = 0
                normalizer = 0

        if args['lr'] > 0 and args['lr_decay'] > 0:
            current_lr = args['lr'] / (1 + (indexs + 1) * args['lr_decay'])
            adjust_learning_rate(optimizer, current_lr)

        if args['lr'] > 0 and args['decay_at_epoch'] > 0 and indexs == args['decay_at_epoch']:
            current_lr = args['lr'] * args['decay_rate']
            adjust_learning_rate(optimizer, current_lr)
            logger.info('lr is modified to: {}'.format(current_lr))

        dev_f1, dev_pre, dev_rec, dev_acc = evaluator.calc_score(model, dev_data.get_tqdm(device, args['batch_size']))

        pw.add_loss_vs_batch({'dev_f1': dev_f1}, indexs, use_logger = True)
        pw.add_loss_vs_batch({'dev_pre': dev_pre, 'dev_rec': dev_rec}, indexs, use_logger = False)
        
        logger.info('Saving model...')
        if dev_f1 > best_f1:
            torch.save(model, os.path.join(args['cp_root'], args['checkpoint_name'], 'best.th'))

        if test_data is not None:
            if dev_f1 > best_f1:
                test_f1, test_pre, test_rec, test_acc = evaluator.calc_score(model, test_data.get_tqdm(device, args['batch_size']))
                best_f1, best_dev_pre, best_dev_rec, best_dev_acc = dev_f1, dev_pre, dev_rec, dev_acc
                pw.add_loss_vs_batch({'test_f1': test_f1}, indexs, use_logger = True)
                pw.add_loss_vs_batch({'test_pre': test_pre, 'test_rec': test_rec}, indexs, use_logger = False)
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args['patience']:
                    break

    pw.close()
