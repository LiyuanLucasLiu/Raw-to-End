
import argparse
import torch_scope
import logging
import random
import json

from betanlp.encoder import strRealTimeEncoderWrapper

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config/twitter_ner.json")
    conf = parser.parse_args()
    with open(conf.config, 'r') as fin:
        args = json.load(fin)

    encoder = strRealTimeEncoderWrapper(args)

    for input_file, save_to in zip(args['input_files'], args['save_to']):

        with open(input_file, 'r') as fin:
            data = json.load(fin)

        x_list = list()
        y_list = list()

        logger.info('Size: {}'.format(len(data)))
        for ins in data:
            x_list.append(ins[0])
            y_list.append(ins[1])

        processed_data, processed_label = encoder(x_list, y_list)

        processed_data['label'] = processed_label
        
        with open(save_to, 'w') as fout:
            json.dump(processed_data, fout)

    x_list = list()
    y_list = list()

    with open(args['input_files'][0], 'r') as fin:
        train_data = json.load(fin)

    for ins in train_data:
        x_list.append(ins[0])
        y_list.append(ins[1])

    with open(args['input_files'][2], 'r') as fin:
        dev_data = json.load(fin)

    for ins in dev_data:
        x_list.append(ins[0])
        y_list.append(ins[1])

    logger.info('Size: {}'.format(len(x_list)))

    processed_data, processed_label = encoder(x_list, y_list)

    processed_data['label'] = processed_label

    with open(args['save_to'][-1], 'w') as fout:
        json.dump(processed_data, fout)
