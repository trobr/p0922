import torch
import torch.nn as nn
from torch.autograd import Function
import posenc_cuda  # the compiled extension

class PosEncFunction(Function):
    @staticmethod
    def forward(ctx, inputs, iteration, kick_in_iter, full_band_iter,
                ):
        # inputs: (B, D)
        outputs = posenc_cuda.forward(inputs, float(iteration),
                                            float(kick_in_iter),
                                            float(full_band_iter),)
        ctx.save_for_backward(inputs)
        # save hyperparams for backward (as ctx attributes)
        ctx.iteration = float(iteration)
        ctx.kick_in_iter = float(kick_in_iter)
        ctx.full_band_iter = float(full_band_iter)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        grad_inputs = posenc_cuda.backward(grad_output, inputs,
                                           float(ctx.iteration),
                                           float(ctx.kick_in_iter),
                                           float(ctx.full_band_iter))
        # None for scalars (not learned or requiring grad)
        return grad_inputs, None, None, None

class EmbedderModule(nn.Module):
    def __init__(self, input_dims=3,
                 kick_in_iter_rate=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.kick_in_iter_rate = kick_in_iter_rate
        self.include_input = False
        # iteration is expected as a scalar passed during forward (e.g., training step)
        # you can change API to store a buffer if desired

    def set_full_band_iter(self, full_band_iter):
        self.full_band_iter = float(full_band_iter)
        self.kick_in_iter = float(self.kick_in_iter_rate * full_band_iter)

    @property
    def out_dim(self):
        return (self.input_dims if self.include_input else 0) + self.num_freqs * 2 * self.input_dims

    def forward(self, inputs, iteration, total_iteration):
        # inputs: (B, input_dims)
        if not inputs.is_cuda:
            raise RuntimeError("EmbedderModule currently requires CUDA tensors")
        # iteration can be a scalar tensor or float
        return PosEncFunction.apply(inputs, iteration, self.kick_in_iter_rate * total_iteration, total_iteration)