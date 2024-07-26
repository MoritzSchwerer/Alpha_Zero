import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Tuple, Optional
from pathlib import Path


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
        broad_cast_every_n: int = 8,
        bottleneck_channels: Optional[int] = 128,
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
        self.broad_cast_every_n = broad_cast_every_n
        self.bottleneck_channels = bottleneck_channels


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


class BasicBlock(nn.Module):
    def __init__(self, make_op, out_channels: int, non_linearity=nn.ReLU()):
        super(BasicBlock, self).__init__()
        self.op = make_op()
        self.bn = nn.BatchNorm2d(out_channels)
        self.non_linearity = non_linearity

    def forward(self, x):
        x = self.op(x)
        x = self.bn(x)
        x = self.non_linearity(x)
        return x


class ResBlockV2(nn.Module):
    def __init__(
        self,
        stack_size: int,
        out_channels: int,
        make_first_op,
        make_inner_op,
        make_last_op,
        bottleneck_channels: Optional[int] = None,
    ):
        super(ResBlockV2, self).__init__()
        if bottleneck_channels is None:
            bottleneck_channels = out_channels
        blocks = []
        for i, op in enumerate(
            [make_first_op] + [make_inner_op] * (stack_size - 2) + [make_last_op]
        ):
            blocks.append(
                BasicBlock(
                    op,
                    out_channels if i == stack_size - 1 else bottleneck_channels,
                    non_linearity=nn.Identity() if i == stack_size - 1 else nn.ReLU(),
                )
            )
        self.main = nn.ModuleList(blocks)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for layer in self.main:
            out = layer(out)
        return self.relu(out + x)


class Broadcast(nn.Module):
    def __init__(self, num_features: int):
        super(Broadcast, self).__init__()
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        assert w == 8 and h == 8

        # x = x.permute(0, 3, 1, 2)
        x = x.view(n, c, h * w)
        x = self.linear(x)
        x = x.view(n, c, h, w)
        # x = x.permute(0, 2, 3, 1)
        return x


class BroadcastResBlock(ResBlockV2):
    def __init__(
        self, make_mix_channel_op, out_channels: int, num_linear_features: int
    ):
        super(BroadcastResBlock, self).__init__(
            stack_size=3,
            out_channels=out_channels,
            make_first_op=make_mix_channel_op,
            make_inner_op=lambda: Broadcast(num_linear_features),
            make_last_op=make_mix_channel_op,
        )


class PredictionNetworkV2(nn.Module):
    def __init__(self, config: NetworkConfig):
        super(PredictionNetworkV2, self).__init__()
        self.num_blocks = config.num_blocks
        self.in_channels = config.in_channels
        self.interm_channels = config.interm_channels
        self.num_linear_features = int(math.prod(config.observation_space[-2:]))
        self.broad_cast_every_n = config.broad_cast_every_n
        self.bottleneck_channels = config.bottleneck_channels
        self.observation_space = config.observation_space
        self.total_num_actions = config.total_num_actions

        blocks = [
            BasicBlock(
                lambda: nn.Conv2d(
                    self.in_channels,
                    self.interm_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                out_channels=self.interm_channels,
            ),
        ]
        for i in range(self.num_blocks):
            if (
                self.broad_cast_every_n > 0
                and i % self.broad_cast_every_n == self.broad_cast_every_n - 1
            ):
                blocks.append(
                    BroadcastResBlock(
                        lambda: nn.Conv2d(
                            self.interm_channels,
                            self.interm_channels,
                            kernel_size=(1, 1),
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        out_channels=self.interm_channels,
                        num_linear_features=self.num_linear_features,
                    )
                )
            elif self.bottleneck_channels is not None:
                blocks.append(
                    ResBlockV2(
                        stack_size=4,
                        out_channels=self.interm_channels,
                        bottleneck_channels=self.bottleneck_channels,
                        make_first_op=lambda: nn.Conv2d(
                            self.interm_channels,
                            self.bottleneck_channels,
                            kernel_size=(1, 1),
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                        make_inner_op=lambda: nn.Conv2d(
                            self.bottleneck_channels,
                            self.bottleneck_channels,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        make_last_op=lambda: nn.Conv2d(
                            self.bottleneck_channels,
                            self.interm_channels,
                            kernel_size=(1, 1),
                            stride=1,
                            padding=0,
                            bias=False,
                        ),
                    )
                )
            else:
                blocks.append(
                    ResBlockV2(
                        stack_size=2,
                        out_channels=self.interm_channels,
                        make_first_op=lambda: nn.Conv2d(
                            self.interm_channels,
                            self.interm_channels,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        make_inner_op=lambda: nn.Conv2d(
                            self.interm_channels,
                            self.interm_channels,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        make_last_op=lambda: nn.Conv2d(
                            self.interm_channels,
                            self.interm_channels,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                    )
                )
        self.main = nn.ModuleList(blocks)
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.interm_channels,
                out_channels=self.interm_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.interm_channels),
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
                bias=False,
            ),
            nn.BatchNorm2d(self.interm_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * self.interm_channels, self.interm_channels),
            nn.ReLU(),
            nn.Linear(self.interm_channels, 3),
        )

    def forward(self, x):
        feats = x
        for i, layer in enumerate(self.main):
            feats = layer(feats)

        policy = self.policy_head(feats)
        value = self.value_head(feats)
        return policy, value


class PredictionNetworkV1(nn.Module):
    def __init__(self, config: NetworkConfig):
        super(PredictionNetworkV1, self).__init__()
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
                bias=not self.use_bn,
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
        # initialize the counter correctly if we already have some networks
        num_files = len(list(self.dir.glob(f"{self.prefix}*.pth")))
        self.counter = num_files + 1

    def save_network(self, net):
        # net = torch.decompile(net)
        self.dir.mkdir(parents=True, exist_ok=True)
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

        net = PredictionNetworkV2(self.network_config)
        if len(files) != 0:
            print("Loading: ", str(files[-1]))
            state_dict = torch.load(files[-1])
            net.load_state_dict(state_dict)

        if not train and self.network_config.half:
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

        # net = torch.compile(net, mode="max-autotune")
        return net


def categorical_to_float(logits: torch.Tensor):
    soft = logits.softmax(-1)
    outcomes = torch.tensor([-1, 0, 1], device=logits.device)
    return torch.sum(soft * outcomes, -1)
