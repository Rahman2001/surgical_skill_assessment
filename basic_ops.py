import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    # def __init__(self, consensus_type, dim=1):
    #     self.consensus_type = consensus_type
    #     self.dim = dim
    #     self.shape = None

    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim):
        shape = input_tensor.size()
        if consensus_type == 'avg':
            output = input_tensor.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        ctx.consensus_type = consensus_type
        ctx.dim = dim
        ctx.shape = shape
        ctx.save_for_backward(output, None, None)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        consensus_type = ctx.consensus_type
        shape = ctx.shape
        dim = ctx.dim

        if consensus_type == 'avg':
            grad_in = grad_output.expand(shape) / float(shape[dim])
        elif consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None, None


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus.apply(input, self.consensus_type, self.dim)
