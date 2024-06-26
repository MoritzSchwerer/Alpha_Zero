import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Tuple
from pathlib import Path


def conv2d(in_channels: int, out_channels: int, bias: bool = False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        bias=bias,
        padding=(1, 1),
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, interm_channels: int, use_bn: bool = True):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.interm_channels = interm_channels
        self.use_bn = use_bn

        self.net = nn.Sequential(
            *[
                conv2d(in_channels, self.interm_channels, bias=not self.use_bn),
                nn.BatchNorm2d(self.interm_channels) if self.use_bn else nn.Identity(),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, use_bn: bool = True):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.use_bn = use_bn

        self.conv1 = conv2d(in_channels, in_channels, bias=not self.use_bn)
        self.bn1 = nn.BatchNorm2d(in_channels) if self.use_bn else nn.Identity()
        self.conv2 = conv2d(in_channels, in_channels, bias=not self.use_bn)
        self.bn2 = nn.BatchNorm2d(in_channels) if self.use_bn else nn.Identity()

    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + inp)


class NetworkConfig:
    def __init__(
        self,
        in_channels: int = 111,
        total_num_actions: int = 4672,
        use_bn: bool = True,
        num_blocks: int = 19,
        interm_channels: int = 256,
        observation_space: Tuple[int, ...] = (111, 8, 8),  # chess
        device: str = "cuda:0",
        channels_last: bool = True,
        half: bool = True,
    ):
        self.in_channels = in_channels
        self.interm_channels = interm_channels
        self.total_num_actions = total_num_actions
        self.use_bn = use_bn
        self.num_blocks = num_blocks
        self.observation_space = observation_space
        self.device = device
        self.channels_last = channels_last
        self.half = half


class PredictionNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super(PredictionNetwork, self).__init__()
        self.in_channels = config.in_channels
        self.interm_channels = config.interm_channels
        self.total_num_actions = config.total_num_actions
        self.use_bn = config.use_bn
        self.num_blocks = config.num_blocks
        self.observation_space = config.observation_space

        self.shared_net = nn.Sequential(
            ConvBlock(self.in_channels, self.interm_channels, self.use_bn),
            *[
                ResBlock(self.interm_channels, self.use_bn)
                for _ in range(self.num_blocks)
            ],
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.interm_channels,
                out_channels=self.interm_channels,
                kernel_size=1,
                bias=not self.use_bn,
            ),
            nn.BatchNorm2d(self.interm_channels) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                math.prod(self.observation_space[-2:]) * self.interm_channels,
                self.total_num_actions,
            ),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.interm_channels,
                out_channels=self.interm_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.interm_channels) if self.use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * self.interm_channels, self.interm_channels),
            nn.ReLU(),
            nn.Linear(self.interm_channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        feats = self.shared_net(x)
        policy = self.policy_head(feats)
        value = self.value_head(feats)
        return policy, value


class NetworkStorage:
    def __init__(self, network_config: NetworkConfig, dir: str, prefix: str = ""):
        self.network_config = network_config
        self.dir = Path(dir)
        self.prefix = prefix
        self.counter = 1

    def save_network(self, net):
        # net = torch.decompile(net)
        net = net.train()
        net = net.to("cpu", memory_format=torch.contiguous_format)
        net = net.float()

        prefix = self.prefix
        if prefix != "":
            prefix = prefix + "_"
        file_name = self.dir / f"{prefix}{str(self.counter).zfill(4)}.pth"

        self.counter += 1
        torch.save(net.state_dict(), file_name)

    def get_latest(
        self,
        train: bool = True,
    ):
        files = list(sorted(self.dir.glob(f"{self.prefix}*.pth")))

        if len(files) == 0:
            net = PredictionNetwork(self.network_config)
        else:
            net = torch.load(files[-1])

        if self.network_config.half:
            net = net.half()

        net = net.to(
            device=self.network_config.device,
            memory_format=(
                torch.channels_last
                if self.network_config.channels_last
                else torch.contiguous_format
            ),
        )
        net = net.train() if train else net.eval()

        # net = torch.compile(mode="max-autotune")
        return net
