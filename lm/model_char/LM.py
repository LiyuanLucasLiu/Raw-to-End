"""
.. module:: LM
    :synopsis: language modeling
 
.. moduleauthor:: Liyuan Liu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_char.utils as utils

class LM(nn.Module):
    """
    The language model model.
    
    Parameters
    ----------
    rnn : ``torch.nn.Module``, required.
        The RNNs network.
    soft_max : ``torch.nn.Module``, required.
        The softmax layer.
    c_num : ``int`` , required.
        The number of words.
    c_dim : ``int`` , required.
        The dimension of word embedding.
    droprate : ``float`` , required
        The dropout ratio.
    label_dim : ``int`` , required.
        The input dimension of softmax.    
    """

    def __init__(self, rnn, c_num, c_dim, t_num, t_dim, droprate):

        super(LM, self).__init__()

        self.rnn = rnn

        self.c_num = c_num
        self.c_dim = c_dim

        self.t_num = t_num
        self.t_dim = t_dim

        self.char_embed = nn.Embedding(c_num, c_dim)
        self.type_embed = nn.Embedding(t_num, t_dim)

        self.rnn_output = self.rnn.output_dim

        self.decode_char = nn.Linear(self.rnn_output, c_num)
        self.decode_type = nn.Linear(self.rnn_output, t_num)

        self.drop = nn.Dropout(p=droprate)

        self.loss = nn.CrossEntropyLoss()

    # def rand_ini(self):
    #     """
    #     Random initialization.
    #     """
    #     self.rnn.rand_ini()
    #     # utils.init_linear(self.project)
    #     self.soft_max.rand_ini()
    #     # if not self.tied_weight:
    #     utils.init_embedding(self.word_embed.weight)

    #     if self.add_proj:
    #         utils.init_linear(self.project)

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.rnn.init_hidden()

    def forward(self, x, t, x_y, t_y):
        """
        Calculate the loss.
        """

        x_emb = self.char_embed(x)
        t_emb = self.type_embed(t)
        
        x_emb = torch.cat([x_emb, t_emb], dim = -1)

        x_emb = self.drop(x_emb)

        rnn_out = self.rnn(x_emb).contiguous().view(-1, self.rnn_output)

        x_decoded = self.decode_char(rnn_out)
        t_decoded = self.decode_type(rnn_out)

        x_loss = self.loss(x_decoded, x_y.view(-1))
        t_loss = self.loss(t_decoded, t_y.view(-1))
        
        return x_loss, t_loss
