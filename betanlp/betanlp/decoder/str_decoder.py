import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import logging
import unicodedata
import torch_scope

from betanlp.common.utils import init_linear, log_sum_exp

logger = logging.getLogger(__name__)

from ipdb import set_trace

class strDecoder(object):

    def __init__(self, arg):

        super(strDecoder, self).__init__()

        logger.info('Building String Decoder')

        if type(arg) is not dict:
            arg = vars(arg)

        try:
            logger.info('Loading label dictionary from: {}'.format(arg['strDecoder']['label_dict']))
            with open(arg['strDecoder']['label_dict'], 'r') as fin:
                self.label_dict = json.load(fin)
        except FileNotFoundError as err:
            logger.error('File not exist: {}'.format(arg['strDecoder']['label_dict']))
            raise

        self.reverse_label_dict = {v: k for k, v in self.label_dict.items()}

        logger.info('str Decoder has been built successfully.')

    def __call__(self, x, y = None):
        return self.forward(x, y)

    def forward(self, x, y):
        """
        decode a sentence in the format of <>
        Parameters
        ----------
        feature: ``list``, required.
            Words list
        label: ``list``, required.
            Label list.
        """
        chunks = ""
        current = None
        
        # set_trace()

        for f, y_ins in zip(x, y):
            label = self.reverse_label_dict[y_ins]

            if label.startswith('B-'):

                if current is not None:
                    chunks += "</"+current+">"
                current = label[2:]
                chunks += "<"+current+">" + f

            elif label.startswith('S-'):

                if current is not None:
                    chunks += "</"+current+">"
                current = label[2:]
                chunks += "<"+current+">" + f + "</"+current+">"
                current = None

            elif label.startswith('I-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f
                    else:
                        chunks += "</"+current+"><"+base+">" + f
                        current = base
                else:
                    current = label[2:]
                    chunks += "<"+current+">" + f

            elif label.startswith('E-'):

                if current is not None:
                    base = label[2:]
                    if base == current:
                        chunks += f + "</"+base+">"
                        current = None
                    else:
                        chunks += "</"+current+"><"+base+">" + f + "</"+base+">"
                        current = None

                else:
                    current = label[2:]
                    chunks += "<"+current+">" + f + "</"+current+">"
                    current = None

            else:
                if current is not None:
                    chunks += "</"+current+">"
                chunks += f
                current = None

        if current is not None:
            chunks += "</"+current+">"

        return chunks
