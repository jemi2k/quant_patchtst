import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        
        """
        this layer is to handle non-stationary data!
        args:
            num_features: number of channels/variables
            eps: small value to prevent division by zero
            affine: if True, learns scale and shift parameters (w and b)
        """
        super().__init__()
        self.eps = eps
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.affine_weight = None
            self.affine_bias = None

        # state placeholders to store stats during the forward pass
        self.mean = None
        self.stdev = None

    def forward(self, x, mode='norm'):
       
        """
        args:
            x: Input tensor of shape [Batch, Time_Steps, Channels]
            mode: 'norm' for scaling input, 'denorm' for unscaling output
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = x - self.mean
            x = x / self.stdev
            if self.affine_weight is not None:
                x = x * self.affine_weight + self.affine_bias
            return x
            
        elif mode == 'denorm':
            if self.mean is None or self.stdev is None:
                raise RuntimeError("you must call forward(mode='norm') before denormalizing.")
            
            # If the output sequence length is different from input sequence length,
            # we need to ensure the dimensions align for broadcasting
            if x.shape[1] != self.mean.shape[1]:
                # We reuse the mean/stdev calculated from the lookback window
                mean = self.mean.repeat(1, x.shape[1], 1)
                stdev = self.stdev.repeat(1, x.shape[1], 1)
            else:
                mean = self.mean
                stdev = self.stdev

            if self.affine_weight is not None:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            
            x = x * stdev
            x = x + mean
            return x

    def _get_statistics(self, x):
        # calculate mean and standard deviation across the Time axis (dim=1)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()