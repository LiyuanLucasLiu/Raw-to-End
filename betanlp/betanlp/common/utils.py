import torch
import numpy as np
import torch.nn as nn

class VariationalDropout(torch.nn.Module):

    def __init__(self, p=0.5):
        super(VariationalDropout, self).__init__()
        self.droprate = p

    def forward(self, x):
        if not self.training or not self.droprate:
            return x

        mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.droprate)
        mask = torch.autograd.Variable(mask, requires_grad=False) / (1 - self.droprate)
        mask = mask.expand_as(x)
        return mask * x

class WordDropout(torch.nn.Module):

    def __init__(self, p=0.05):
        super(WordDropout, self).__init__()
        self.droprate = p

    def forward(self, x):
        if not self.training or not self.droprate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.droprate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

def iob_iobes(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.startswith('O'):
            new_tags.append(tag)
        elif tag.startswith('B-'):
            if i + 1 < len(tags) and tags[i + 1].startswith('I-'):
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        else:
            assert tag.startswith('I-')
            if i + 1 < len(tags) and tags[i + 1].startswith('I-'):
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))

    return new_tags

def repackage_hidden(h, device = None):
    """
    Wraps hidden states in new Variables, to detach them from their history

    Parameters
    ----------
    h : ``Tuple`` or ``Tensors``, required.
        Tuple or Tensors, hidden states.

    Returns
    -------
    hidden: ``Tuple`` or ``Tensors``.
        detached hidden states
    """
    if type(h) == torch.Tensor:
        if device is None:
            return h.detach()
        else:
            return h.detach().to(device)
    else:
        return tuple(repackage_hidden(v) for v in h)
        
def init_linear(input_linear):
    """
    random initialize linear projection.
    """
    # bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    # nn.init.uniform_(input_linear.weight, -bias, bias)
    # if input_linear.bias is not None:
    #     input_linear.bias.data.zero_()
    return

def log_sum_exp(vec):
    max_score, _ = torch.max(vec, 1)

    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.unsqueeze(1).expand_as(vec)), 1))

def adjust_learning_rate(optimizer, lr):
    """
    adjust learning to the the new value.
    Parameters
    ----------
    optimizer : required.
        pytorch optimizer.
    float :  ``float``, required.
        the target learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
