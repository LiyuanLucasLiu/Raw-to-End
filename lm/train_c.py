from __future__ import print_function
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import pickle
import math
import argparse
import json
import os
import sys
import itertools
import functools

from torch_scope import wrapper
from ipdb import set_trace

from model_char.LM import LM
from model_char.basic import BasicRNN
from model_char.dataset import LargeDataset
import model_char.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="auto")
    parser.add_argument('--cp_root', default='./checkpoint')
    parser.add_argument('--checkpoint_name', default='c0')
    parser.add_argument('--git_tracking', action='store_true')
    parser.add_argument('--spreadsheet_name', type=str, default='TwitterNER')
    parser.add_argument('--description', type=str, default='character level language model')

    parser.add_argument('--restore_checkpoint', default='')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--dataset_folder', default='./encoded_tweets/')
    parser.add_argument('--input_dict', default='./enriched_char_dict.json')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sequence_length', type=int, default=256)
    parser.add_argument('--type_dim', type=int, default=128)
    parser.add_argument('--char_dim', type=int, default=128)
    parser.add_argument('--hid_dim', type=int, default=2048)
    parser.add_argument('--layer_num', type=int, default=1)
    parser.add_argument('--droprate', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--clip', type=float, default=5)
    parser.add_argument('--update', choices=['Adam', 'Adagrad', 'Adadelta'], default='Adam', help='adam is the best')
    parser.add_argument('--rnn_layer', choices=['Basic'], default='Basic')
    parser.add_argument('--rnn_unit', choices=['gru', 'lstm', 'rnn'], default='lstm')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--epoch_size', type=int, default=4000)
    parser.add_argument('--patience', type=float, default=10)
    args = parser.parse_args()

    pw = wrapper(os.path.join(args.cp_root, args.checkpoint_name), args.checkpoint_name, checkpoints_to_keep = 3, enable_git_track=args.git_tracking)
    pw.set_level('info')
    pw.add_description(args.description)

    gpu_index = pw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)

    pw.info('Loading dataset.')
    with open(args.input_dict, 'r') as fin:
        all_dict = json.load(fin)

        lm_char_dict = all_dict['lm_char_dict']
        lm_type_dict = all_dict['lm_type_dict']

    train_loader = LargeDataset(args.dataset_folder, args.batch_size, args.sequence_length, args.reverse)
    
    pw.info('Building models.')
    rnn_map = {'Basic': BasicRNN}
    rnn_layer = rnn_map[args.rnn_layer](args.layer_num, args.rnn_unit, args.char_dim + args.type_dim, args.hid_dim, args.droprate)
    
    lm_model = LM(rnn_layer, len(lm_char_dict), args.char_dim, len(lm_type_dict), args.type_dim, args.droprate)
    
    pw.info('Building optimizer.')
    optim_map = {'Adam' : optim.Adam, 'Adagrad': optim.Adagrad, 'Adadelta': optim.Adadelta}
    if args.lr > 0:
        optimizer=optim_map[args.update](lm_model.parameters(), lr=args.lr)
    else:
        optimizer=optim_map[args.update](lm_model.parameters())

    if args.restore_checkpoint:
        if os.path.isfile(args.restore_checkpoint):
            pw.info("loading checkpoint: '{}'".format(args.restore_checkpoint))
            model_file = wrapper.restore_checkpoint(args.restore_checkpoint)['model']
            lm_model.load_state_dict(model_file, False)
        elif os.path.isdir(args.restore_checkpoint):
            model_file, file_list = utils.checkpoint_average(args.restore_checkpoint)
            for file_ins in file_list:
                pw.info("averaging checkpoint includes: '{}'".format(file_ins))
            lm_model.load_state_dict(model_file, False)
        else:
            pw.info("no checkpoint found at: '{}'".format(args.restore_checkpoint))
    lm_model.to(device)

    pw.info('Saving configues.')
    pw.save_configue(args)

    pw.info('Setting up training environ.')
    best_train_ppl = float('inf')
    cur_lr = args.lr
    batch_index = 0
    epoch_x_loss = 0
    patience = 0

    try:
        for indexs in range(args.epoch):
    
            pw.info('############')
            pw.info('Epoch: {}'.format(indexs))
            pw.nvidia_memory_map()

            lm_model.train()

            for word_t, type_t, label_word_t, label_type_t in train_loader.get_tqdm(device):
                # set_trace()
                lm_model.zero_grad()
                x_loss, t_loss = lm_model(word_t, type_t, label_word_t, label_type_t)
                
                loss = x_loss + t_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm_model.parameters(), args.clip)
                optimizer.step()

                batch_index += 1
                if 0 == batch_index % args.interval:
                    s_x_loss = utils.to_scalar(x_loss)
                    s_t_loss = utils.to_scalar(t_loss)
                    pw.add_loss_vs_batch({'batch_x_loss': s_x_loss, 'batch_t_loss': s_t_loss}, batch_index, use_logger = False, use_sheet_tracker = True)

                epoch_x_loss += utils.to_scalar(x_loss)

                if 0 == batch_index % args.epoch_size:
                    epoch_ppl = math.exp(epoch_x_loss / args.epoch_size)
                    pw.add_loss_vs_batch({'train_ppl': epoch_ppl}, batch_index, use_sheet_tracker = True)
                    
                    if epoch_ppl < best_train_ppl:
                        best_train_ppl = epoch_ppl
                        patience = 0
                    else:
                        patience += 1
                    
                    epoch_x_loss = 0
                    pw.save_checkpoint(model = lm_model, optimizer = optimizer, is_best = False)

                if patience > args.patience and cur_lr > 0:
                    patience = 0
                    cur_lr *= args.lr_decay
                    best_train_ppl = float('inf')
                    pw.info('adjust_learning_rate...')
                    utils.adjust_learning_rate(optimizer, cur_lr)

            pw.save_checkpoint(model = lm_model, optimizer = optimizer, is_best = True)

    except Exception as e_ins:

        pw.info('Exiting from training early')
        pw.save_checkpoint(model = lm_model, optimizer = optimizer, is_best = False)

        print(type(e_ins))
        print(e_ins.args)
        print(e_ins)

    pw.close()
