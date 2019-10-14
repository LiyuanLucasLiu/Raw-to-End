"""
.. module:: basic
    :synopsis: basic rnn
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from betanlp.common.utils import repackage_hidden

class BasicUnit(nn.Module):
    """
    The basic recurrent unit for the vanilla stacked RNNs.

    Parameters
    ----------
    unit : ``str``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, unit, input_dim, hid_dim, droprate):
        super(BasicUnit, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.layer = rnnunit_map[unit](input_dim, hid_dim, 1)

        self.droprate = droprate

        self.output_dim = hid_dim

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.hidden_state = None

    # def rand_ini(self):
    #     """
    #     Random Initialization.
    #     """
    #     return

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.   
            The output of RNNs.
        """
        if not self.training or random.uniform(0, 1) < 0.2:
            self.init_hidden()
        elif self.hidden_state is not None:
            device = next(self.parameters()).device
            self.hidden_state = repackage_hidden(self.hidden_state, device)

        # if self.hidden_state is not None:
        #     device = next(self.parameters()).device
        #     self.hidden_state = repackage_hidden(self.hidden_state, device)

        out, new_hidden = self.layer(x, self.hidden_state)

        self.hidden_state = new_hidden

        return out

class BasicRNN(nn.Module):
    """
    The multi-layer recurrent networks for the vanilla stacked RNNs.

    Parameters
    ----------
    layer_num: ``int``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(BasicRNN, self).__init__()

        layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate)] + [BasicUnit(unit, hid_dim, hid_dim, droprate) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*layer_list)
        self.output_dim = layer_list[-1].output_dim

        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer.children():
            tup.init_hidden()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        return self.layer(x)
        