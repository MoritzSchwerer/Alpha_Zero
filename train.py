import multiprocessing as mp
import torch

from config import AlphaZeroConfig
from storage import ReplayBuffer
from gumbel_alpha_zero import play_game
from network import NetworkStorage

from tqdm import tqdm


class Trainer:
    """
    this class sets up all the dependencies
    and runs both self-play as well as the
    training of the network using the self-
    play samples
    """

    def __init__(
        self,
        config: AlphaZeroConfig,
        replay_buffer: ReplayBuffer,
        network_storage: NetworkStorage,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.network_storage = network_storage

        if not self.replay_buffer.init_from_dataset():
            self.replay_buffer.create_dataset()

    def self_play(self):
        """
        self play get's the latest network and runs self play
        with that to produce new games
        """
        network = self.network_storage.get_latest()
        games_per_run = self.config.num_processes * self.config.self_play_batch_size
        num_iterations = (
            self.replay_buffer.config.file_size
            - self.replay_buffer.current_size % self.replay_buffer.config.file_size
        ) // games_per_run
        print(f"Starting with {games_per_run} games per iteration")
        print(f"and {num_iterations} iterations")
        print(f"Totaling {games_per_run*num_iterations} games.")
        for i in range(num_iterations):
            print("=" * 88)
            print(f"Starting iteration {i+1}/{num_iterations}")
            pool = mp.get_context("spawn").Pool(processes=self.config.max_num_threads)
            try:
                games = pool.starmap(
                    play_game, [(self.config, network)] * self.config.num_processes
                )
                games = [item for sublist in games for item in sublist]
            finally:
                pool.close()
                pool.join()

            # this is to make sure that we only fill up a file exactly
            file_size =self.replay_buffer.config.file_size
            file_games_left = file_size - self.replay_buffer.current_size % file_size
            games = games[:file_games_left]

            self.replay_buffer.append(games)

    # TODO: write this
    def train(self):
        network = self.network_storage.get_latest(train=True)
        optim = torch.optim.AdamW(network.parameters(), lr=0.2)

        for i in tqdm(range(1000)):
            data = self.replay_buffer.sample_examples(batch_size=4096)
            state, value_target, policy_target = data
            policy_pred, value_pred = network(state)

            # TODO: place holder
            loss = (value_pred - value_target) + (policy_pred - policy_target)

            optim.zero_grad()
            loss.backward()
            optim.step()
