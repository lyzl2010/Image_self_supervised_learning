import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        print('memory',memory.size())
        print('params',params.size())
        print('x',x.size())
        print('y',y.size())
        T = params[0]
        batchSize = x.size(0)
        outputSize = memory.size(1)
        inputSize = memory.size(1)

        # inner product
        out = torch.mm(x, memory.t())
        out.div_(T) # batchSize * N
        
        self.data_for_backward = x, memory, y, params 

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.data_for_backward
        batchSize = gradOutput.size(0)
        T = params[0]
        momentum = params[1]
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return Variable(gradInput), None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.Tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

