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
import pickle
import random
from tqdm import tqdm

from torch.utils.data import Dataset

from ipdb import set_trace

# class EvalDataset(object):
#     """    
#     Dataset for Language Modeling
#     Parameters
#     ----------
#     dataset : ``list``, required.
#         The encoded dataset (outputs of preprocess scripts).
#     sequence_length: ``int``, required.
#         Sequence Length.
#     """
#     def __init__(self, dataset, sequence_length):
#         super(EvalDataset, self).__init__()
#         self.dataset = dataset

#         self.sequence_length = sequence_length

#         self.construct_index()

#     def get_tqdm(self, device):
#         """
#         construct dataset reader and the corresponding tqdm.

#         Parameters
#         ----------
#         device: ``torch.device``, required.
#             the target device for the dataset loader.

#         """
#         return tqdm(self.reader(device), mininterval=2, total=self.index_length, leave=False, file=sys.stdout, ncols=80)

#     def construct_index(self):
#         """
#         construct index for the dataset.
#         """
#         token_per_batch = self.sequence_length
#         tot_num = len(self.dataset) - 1
#         res_num = tot_num - tot_num % token_per_batch

#         self.x = list(torch.unbind(torch.LongTensor(self.dataset[0:res_num]).view(-1, self.sequence_length), 0))
#         self.y = list(torch.unbind(torch.LongTensor(self.dataset[1:res_num+1]).view(-1, self.sequence_length), 0))

#         self.x.append(torch.LongTensor(self.dataset[res_num:tot_num]))
#         self.y.append(torch.LongTensor(self.dataset[res_num+1:tot_num+1]))

#         self.index_length = len(self.x)
#         self.cur_idx = 0

#     def reader(self, device):
#         """
#         construct dataset reader.

#         Parameters
#         ----------
#         device: ``torch.device``, required.
#             the target device for the dataset loader.

#         Returns
#         -------
#         reader: ``iterator``.
#             A lazy iterable object        
#         """
#         if self.cur_idx == self.index_length:
#             self.cur_idx = 0
#             raise StopIteration

#         word_t = self.x[self.cur_idx].to(device).view(-1, 1)
#         label_t = self.y[self.cur_idx].to(device).view(-1, 1)

#         self.cur_idx += 1
        
#         yield word_t, label_t

class LargeDataset(object):
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
    def __init__(self, root, batch_size, sequence_length, reverse = False):
        super(LargeDataset, self).__init__()
        self.root = root

        self.file_list = list()
        list_dirs = os.walk(root)
        for root, dirs, files in list_dirs:
            for file in files:
                self.file_list.append(os.path.join(root, file))
        self.shuffle()

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
        previous_text = list()
        previous_type = list()

        for file in self.file_list:
            x, t, x_y, t_y, previous_text, previous_type = self.open_next(file, previous_text, previous_type)
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

    def open_next(self, file, previous_text = list(), previous_type = list()):
        """
        Open the next file.
        """
        dataset = pickle.load(open(file, 'rb'))

        if self.reverse:
            text_array = previous_text + dataset['text_array'][::-1]
            type_array = previous_type + dataset['type_array'][::-1]
        else:
            text_array = previous_text + dataset['text_array']
            type_array = previous_type + dataset['type_array']

        res_num = len(text_array) - 1
        if res_num < self.token_per_batch:
            return None, None, None, None, text_array, type_array

        res_num = res_num - res_num % self.token_per_batch

        x = torch.LongTensor(text_array[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
        t = torch.LongTensor(type_array[0:res_num]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
        x_y = torch.LongTensor(text_array[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()
        t_y = torch.LongTensor(type_array[1:res_num+1]).view(self.batch_size, -1, self.sequence_length).transpose_(0, 1).transpose_(1, 2).contiguous()

        return x, t, x_y, t_y, text_array[res_num + 1: ], type_array[res_num + 1: ]
