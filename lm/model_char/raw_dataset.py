"""
.. module:: dataset
    :synopsis: dataset for language modeling
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

import unicodedata
import random
from tqdm import tqdm

import emoji

from torch.utils.data import Dataset

from ipdb import set_trace

class LargeRawDataset(object):
    """    
    Lazy Dataset for Language Modeling

    Parameters
    ----------
    root : ``str``, required.
        The root folder for dataset files.
    range_idx : ``int``, required.
        The maximum file index for the input files (train_*.pk).
    batch_size : ``int``, required.
        Batch size.
    sequence_length: ``int``, required.
        Sequence Length.
    """
    def __init__(self, root, batch_size, sequence_length, char_dict, type_dict, reverse = False):
        super(LargeRawDataset, self).__init__()
        self.root = root

        self.file_list = list()
        list_dirs = os.walk(root)
        for root, dirs, files in list_dirs:
            for file in files:
                self.file_list.append(os.path.join(root, file))
        self.shuffle()

        self.char_dict = char_dict
        self.type_dict = type_dict
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.token_per_batch = self.batch_size * self.sequence_length

        self.reverse = reverse

        self.total_batch_num = -1

    def shuffle(self):
        """
        shuffle dataset
        """
        random.shuffle(self.file_list)

    def get_tqdm(self, device):
        """
        construct dataset reader and the corresponding tqdm.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.        
        """
        if self.total_batch_num <= 0:
            return tqdm(self.reader(device), mininterval=2, leave=False, file=sys.stdout).__iter__()
        else:
            return tqdm(self.reader(device), mininterval=2, total=self.total_batch_num, leave=False, file=sys.stdout, ncols=80).__iter__()

    def reader(self, device):
        """
        construct dataset reader.

        Parameters
        ----------
        device: ``torch.device``, required.
            the target device for the dataset loader.

        Returns
        -------
        reader: ``iterator``.
            A lazy iterable object        
        """
        batch_count = 0

        for x, t, x_y, t_y in self.raw_text_reader():

            if x is None:
                continue

            index_length = x.size(0)

            for cur_idx in range(index_length):

                word_t = x[cur_idx].to(device)
                type_t = t[cur_idx].to(device)
                label_word_t = x_y[cur_idx].to(device)
                label_type_t = t_y[cur_idx].to(device)

                yield word_t, type_t, label_word_t, label_type_t

            batch_count += index_length

        self.total_batch_num = batch_count
        self.shuffle()

    def raw_text_reader(self):
        """
        Open the next file.
        """
        previous_text = list()
        previous_type = list()

        lm_char_type_dict = dict()
        for k, v in self.char_dict.items():
            if k.islower():
                lm_char_type_dict[k] = self.type_dict['<low>']
            elif k.isdigit():
                lm_char_type_dict[k] = self.type_dict['<num>']
            else:
                lm_char_type_dict[k] = self.type_dict['<pun>']

        dataset_type = list()
        dataset_text = list()

        for file in self.file_list:
            for line in open(file, 'r'):
                line = line.rstrip()

                for char in line:
                    if char in self.char_dict:
                        dataset_text.append(self.char_dict[char])
                        dataset_type.append(lm_char_type_dict[char])
                    elif char.isupper() and char.lower() in self.char_dict:
                        dataset_text.append(self.char_dict[char.lower()])
                        dataset_type.append(self.type_dict['<up>'])
                    elif char in emoji.UNICODE_EMOJI:
                        dataset_text.append(self.char_dict['<emoji>'])
                        dataset_type.append(self.type_dict['<pun>'])
                    else:
                        char_list = unicodedata.normalize('NFKD', char)

                        for char_tup in char_list:
                            if char_tup in self.char_dict:
                                dataset_text.append(self.char_dict[char_tup])
                                dataset_type.append(lm_char_type_dict[char_tup])
                            elif char_tup.isupper():
                                char_tup = char_tup.lower()
                                if char in self.char_dict:
                                    dataset_text.append(self.char_dict[char])
                                else:
                                    dataset_text.append(self.char_dict['<unk>'])
                                dataset_type.append(self.type_dict['<up>'])
                            elif char_tup.islower():
                                dataset_text.append(self.char_dict['<unk>'])
                                dataset_type.append(self.type_dict['<low>'])
                            elif char_tup.isdigit():
                                dataset_text.append(self.char_dict['<unk>'])
                                dataset_type.append(self.type_dict['<num>'])
                            else:
                                dataset_text.append(self.char_dict['<unk>'])
                                dataset_type.append(self.type_dict['<pun>'])

                dataset_text.append(self.char_dict['<eof>'])
                dataset_type.append(self.type_dict['<pun>'])

                if len(dataset_text) > 200000:

                    if self.reverse:
                        text_array = previous_text + dataset_text[::-1]
                        type_array = previous_type + dataset_type[::-1]
                    else:
                        text_array = previous_text + dataset_text
                        type_array = previous_type + dataset_type

                    res_num = len(text_array) - 1
                    if res_num < self.token_per_batch:
                        return None, None, None, None, text_array, type_array

                    res_num = res_num - res_num % self.token_per_batch

                    x = torch.LongTensor(text_array[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
                    t = torch.LongTensor(type_array[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
                    x_y = torch.LongTensor(text_array[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
                    t_y = torch.LongTensor(type_array[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
                    
                    previous_text, previous_type = text_array[res_num + 1: ], type_array[res_num + 1: ]

                    dataset_type = list()
                    dataset_text = list()

                    yield x, t, x_y, t_y
