#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code base on https://pytorch.org/docs/stable/notes/extending.html
and on CustomizedLinearLayer from https://github.com/paulmorio/gincco
Torch NN Layer which uses a mask to simulate customized computational graphs
"""
import math
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

#################################
# Define custom autograd function for masked connection.

class BioLayerMaskFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class MaskedBioLayer(nn.Module):
    """MaskedBioLayer operate as a linear layer, but
        with masked connections"""

    def __init__(self, mask: torch.Tensor, bias:bool=True, init_weights:torch.Tensor=None):
        """
        Arguments
        ------------------
        mask [torch.Tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected
        
        bias [bool]:
            flag of bias.

        init_weights [torch.Tensor]:
            initial weight matrix
            the shape has to be euqal to that of the mask transposed
        
        """
        super(MaskedBioLayer, self).__init__()

        if init_weights is not None: assert init_weights.T.shape == mask.shape, (init_weights.shape, mask.shape)

        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        self.init_weights = init_weights

        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask.to(device), requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.

        self.weight = nn.Parameter(torch.ones(self.output_features, self.input_features).to(device))

        if bias:
            self.bias = nn.Parameter(torch.ones(self.output_features).to(device))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        if self.init_weights is not None:
            self.weight = nn.Parameter(self.init_weights)
        else:
            # stdv = 1. / math.sqrt(self.weight.size(1))
            # self.weight.data.uniform_(-stdv, stdv)
            self.weight = nn.Parameter(torch.ones_like(self.mask))

        
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data = nn.Parameter(torch.ones_like(self.bias.data).to(device))


    def forward(self, input):
        ## Apply the mask
        return BioLayerMaskFunction.apply(input.float(), self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


if __name__ == 'check grad':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    customlinear = BioLayerMaskFunction.apply

    input = (
            torch.randn(20,20,dtype=torch.double,requires_grad=True),
            torch.randn(30,20,dtype=torch.double,requires_grad=True),
            None,
            None,
            )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)