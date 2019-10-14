import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import json
import logging
import torch_scope

from betanlp.decoder import spCRFDecoder, strDecoder
from betanlp.encoder import spEncoderWrapper, denRNNEncoder

from ipdb import set_trace

logger = logging.getLogger(__name__)

class seqLabel(nn.Module):

    def __init__(self, arg):

        super(seqLabel, self).__init__()

        self.spEncoder = spEncoderWrapper(arg)
        self.denEncoder = denRNNEncoder(arg)
        self.spDecoder = spCRFDecoder(arg)
        self.strDecoder = strDecoder(arg)

    def forward(self, x, y=None):
        sp_out = self.spEncoder(x)
        den_out = self.denEncoder(sp_out)
        crf_out = self.spDecoder(den_out, y)

        return crf_out

    def decode(self, x, y):
        return self.strDecoder(x, y)

class seqLabelEvaluator(object):

    def __init__(self, decoder):
        self.decoder = decoder
        self.reset()

    def reset(self):
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def calc_f1_batch(self, decoded_data, target_data):
        decoded_data = decoded_data['label'].cpu()
        batch_decoded = torch.unbind(decoded_data, 0)

        for decoded, target in zip(batch_decoded, target_data):
            
            target = target['label']
            length = len(target)
            best_path = decoded[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(best_path.numpy(), target)
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def calc_acc_batch(self, decoded_data, target_data):
        decoded_data = decoded_data['label'].cpu()
        batch_decoded = torch.unbind(decoded_data, 0)

        for decoded, target in zip(batch_decoded, target_data):
            
            target = target['label']
            length = len(target)
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def f1_score(self):

        if self.guess_count == 0:
            return 0.0, 0.0, 0.0, 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        return f, precision, recall, accuracy

    def acc_score(self):

        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy        

    def eval_instance(self, best_path, gold):

        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        gold_chunks = self.decoder(gold)
        gold_count = len(gold_chunks)

        guess_chunks = self.decoder(best_path)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def calc_score(self, seq_model, dataset_loader):

        seq_model.eval()
        self.reset()

        for x, y in dataset_loader:
            decoded = seq_model(x)
            # self.eval_b(decoded, y)
            # set_trace()
            self.calc_f1_batch(decoded, y)

        # return self.calc_s()
        return self.f1_score()
