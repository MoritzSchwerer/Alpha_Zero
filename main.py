import torch
import multiprocessing as mp
import os
import time


# from pettingzoo.classic import chess_v6
from network import PredictionNetwork
from alpha_zero import AlphaZeroConfig, play_game
from game import Chess

"""
Took 89   seconds for 48  games (1.9 per game)
"""


def new_game():
    game = Chess()
    game.reset()
    return game


def main_multi():
    start = time.time_ns()

    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    num_threads = 6
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
    res = [res.get() for res in results]

    results = []
    for r in res:
        results.extend(r)
    print(len(results))
    for r in results:
        print(r)
    print(
        f'{num_threads*config.self_play_batch_size} games took {(time.time_ns() - start) // 1e9} seconds.'
    )


def main():
    start = time.time_ns()
    config = AlphaZeroConfig(new_game)

    net = PredictionNetwork(
        in_channels=111,
        use_bn=False,
    ).eval()

    torch.compile(net, mode='max-autotune')
    torch.backends.cudnn.benchmark = True
    history = play_game(config, net)
    print(
        f'{config.self_play_batch_size} games took {(time.time_ns() - start) // 1e9} seconds.'
    )


if __name__ == '__main__':
    main_multi()
