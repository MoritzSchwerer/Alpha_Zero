import torch
import multiprocessing as mp
import os
import time


from network import PredictionNetwork
from gumbel_alpha_zero import play_game
from game import Chess
from config import AlphaZeroConfig

"""
Took 89   seconds for 48   games (1.9 per game)
Took 1010 seconds for 1024 games ()
Took 846  seconds for 768  games ()
"""


def new_game():
    game = Chess()
    game.reset()
    return game


def main():
    start = time.time_ns()
    # os.environ['MKL_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)
    config = AlphaZeroConfig(new_game)

    net = PredictionNetwork(
        in_channels=111,
        use_bn=False,
    ).eval()

    torch.compile(net, mode='max-autotune')
    torch.backends.cudnn.benchmark = True
    history = play_game(config, net)
    print(len(history))
    print(
        f'{config.self_play_batch_size} games took {(time.time_ns() - start) // 1e9} seconds.'
    )


if __name__ == '__main__':
    main()
