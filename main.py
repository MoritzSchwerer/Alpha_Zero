import torch
import multiprocessing as mp
import os
import time


# from pettingzoo.classic import chess_v6
from network import PredictionNetwork
from alpha_zero import AlphaZeroConfig, play_game, play_game_gumbel
from game import Chess

"""
Took 89   seconds for 48   games (1.9 per game)
Took 1010 seconds for 1024 games ()
Took 846  seconds for 768  games ()
"""


def new_game():
    game = Chess()
    game.reset()
    return game


def main_multi():
    start = time.time_ns()

    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    config = AlphaZeroConfig(new_game)

    networks = [
        PredictionNetwork(
            in_channels=111,
            use_bn=False,
        ).eval()
        for _ in range(config.num_processes)
    ]
    for net in networks:
        torch.compile(net, mode='max-autotune')
    torch.backends.cudnn.benchmark = True

    with mp.get_context('spawn').Pool(
        processes=config.max_num_threads
    ) as pool:
        results = pool.starmap(
            play_game_gumbel, [(config, net) for net in networks]
        )
        pool.close()
        pool.join()

    print(len(results))
    for r in results:
        print(r.outcome)
    print(
        f'Took {str((time.time_ns() - start) // 1e9).ljust(4)} seconds for {str(config.num_processes).ljust(4)} games.'
    )


def main():
    start = time.time_ns()
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    config = AlphaZeroConfig(new_game)

    net = PredictionNetwork(
        in_channels=111,
        use_bn=False,
    ).eval()

    torch.compile(net, mode='max-autotune')
    torch.backends.cudnn.benchmark = True
    history = play_game_gumbel(config, net)
    print(len(history))
    print(
        f'{config.self_play_batch_size} games took {(time.time_ns() - start) // 1e9} seconds.'
    )


if __name__ == '__main__':
    main_multi()
