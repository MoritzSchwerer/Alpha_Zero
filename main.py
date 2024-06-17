import torch
import multiprocessing as mp
import os


# from pettingzoo.classic import chess_v6
from network import PredictionNetwork
from alpha_zero import AlphaZeroConfig, play_game
from game import Chess

"""
Took 14.1 minutes for 32 * 6 games
"""


def new_game():
    game = Chess()
    game.reset()
    return game


def main_multi():
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    num_threads = 1
    config = AlphaZeroConfig(new_game)

    networks = [
        PredictionNetwork(
            in_channels=111,
            use_bn=False,
        ).eval()
        for _ in range(num_threads)
    ]
    for net in networks:
        torch.compile(net, mode='max-autotune')
    torch.backends.cudnn.benchmark = True

    with mp.get_context('spawn').Pool(processes=num_threads) as pool:
        results = [
            pool.apply_async(play_game, (config, net)) for net in networks
        ]
        pool.close()
        pool.join()
    results = [res.get() for res in results]
    print(len(results))
    print(results[0])


def main():
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    config = AlphaZeroConfig(new_game)

    net = PredictionNetwork(
        in_channels=111,
        use_bn=False,
    ).eval()

    torch.compile(net, mode='max-autotune')
    torch.backends.cudnn.benchmark = True
    history = play_game(config, net)


if __name__ == '__main__':
    main()
