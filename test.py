import torch
from network import PredictionNetwork

for _ in range(100):
    net = PredictionNetwork(111, use_bn=False)
    input = torch.rand(1, 111, 8, 8)
    out = net(input)
    print(out.shape)
