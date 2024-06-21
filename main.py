import torch
import multiprocessing as mp
import os
import time

from network import PredictionNetwork
from gumbel_alpha_zero import play_game
from game import Chess
from config import AlphaZeroConfig


"""
Took 89   seconds for 48   games (1.9s per game)
Took 1010 seconds for 1024 games (1.0s per game)
Took 846  seconds for 768  games (1.2s per game)
"""

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def main():
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    config = AlphaZeroConfig()
    num_processes = 64
    net = PredictionNetwork(
        in_channels=111,
        use_bn=False,
        interm_channels=128,
        # num_blocks=39,
    ).eval()

    if torch.cuda.is_available() and torch.backends.cudnn.version() >= 7603:
        net = net.to(device=DEVICE, memory_format=torch.channels_last).half()

    torch.compile(net, mode='max-autotune')
    start = time.time_ns()
    with mp.get_context('spawn').Pool(processes=4) as pool:
        games = pool.starmap(play_game, [(config, net)] * num_processes)
    # for _ in range(10):
    #     game = play_game(config, net)
    print(
        f'Took {round((time.time_ns()-start) / 1e9, 1)} seconds for {num_processes} games'
    )


if __name__ == '__main__':
    main()
