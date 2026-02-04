import torch
import torch.nn as nn
import elasticdeform.torch as etorch

class ElasticDeformation(nn.Module):
    def __init__(self, sigma=10, points=3):
        super().__init__()
        self.sigma = sigma
        self.points = points

    def forward(self, *inputs):
        if not inputs:
            return inputs
        
        device = inputs[0].device
        # Generate random displacement (2, 3, 3) * sigma
        displacement = torch.randn(2, self.points, self.points, device=device) * self.sigma
        
        # Prepare inputs for elasticdeform
        # It expects a list of tensors
        img_inputs = list(inputs)
        # orders = []
        assert len(img_inputs) == 2
        orders = [3, 0]
        
        # Apply deformation
        # axis=(1, 2) assumes inputs are (C, H, W) and we deform H, W
        outputs = etorch.deform_grid(img_inputs, displacement, order=orders, axis=(1, 2))
        
        if len(inputs) == 1:
            return outputs[0]
        return tuple(outputs)
