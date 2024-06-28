import torch
import torch.nn.functional as F
import concurrent.futures
import multiprocessing as mp
import time


from torch.amp import autocast
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
        beta_value: float = 0.5,
        lr: float = 0.2,
        train_factor: int = 4,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.network_storage = network_storage
        self.beta_value = beta_value
        self.current_lr = lr
        self.train_factor = train_factor

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
            try:
                executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.config.max_num_threads, mp_context=mp.get_context('spawn'))
                futures = []
                for _ in range(self.config.num_processes):
                    futures.append(executor.submit(play_game, self.config, self.network_storage.network_config, network))

                pbar = tqdm(total=self.config.num_processes)
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    games = future.result()

                    # this is to make sure that we only fill up a file exactly
                    file_size =self.replay_buffer.config.file_size
                    file_games_left = file_size - self.replay_buffer.current_size % file_size
                    games = games[:file_games_left]

                    self.replay_buffer.append(games)
                    del games
                executor.shutdown(wait=True)
            except Exception as e:
                print(e)
                executor.shutdown()
            finally:
                executor.shutdown()


    # TODO: track stats here like losses and so on
    def train(self):
        network = self.network_storage.get_latest(train=True)
        optim = torch.optim.AdamW(network.parameters(), lr=self.current_lr)

        batch_size = self.config.batch_size
        num_iterations = (self.replay_buffer.current_size * self.train_factor) // batch_size

        print(f"Running training for {num_iterations} iterations")
        print(f"making {num_iterations*batch_size} samples out of {self.replay_buffer.current_size} games")
        pbar = tqdm(total=num_iterations)
        total_time_sample = 0
        total_time_network = 0
        for i in range(num_iterations):

            start_time = time.time_ns()
            data = self.replay_buffer.sample_examples(batch_size=batch_size)
            total_time_sample += (time.time_ns()-start_time)

            state, value_target, policy_target = data
            value_target = torch.from_numpy(value_target).to(device='cuda')
            policy_target = torch.from_numpy(policy_target).to(device='cuda')
            state = (
                torch.from_numpy(state)
                .reshape(-1, 8, 8, 111)
                .permute(0, 3, 1, 2)
                .to(
                    device='cuda',
                    memory_format=torch.channels_last,
                    dtype=torch.float16,
                )
            )
            start_time = time.time_ns()
            with autocast(device_type='cuda'):
                policy_pred, value_pred = network(state)
                policy_pred = torch.log_softmax(policy_pred, 1)
                imp_policy_target = torch.log_softmax(policy_pred.detach() + transform(policy_target), 1)

                value_loss = (value_target - value_pred.view(-1)).pow(2).mean(0)
                policy_loss = F.kl_div(policy_pred, imp_policy_target, reduction='batchmean', log_target=True)

                loss = self.beta_value * value_loss + policy_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_time_network += (time.time_ns()-start_time)
            sample_time = int(total_time_sample / (i+1) / 1e6)
            network_time = int(total_time_network / (i+1) / 1e6) 
            losses = (str(round(value_loss.item(), 3)).ljust(5), str(round(policy_loss.item(), 3)).ljust(5))
            pbar.set_postfix({'sample': sample_time, 'losses': losses})
            pbar.update(1)

        self.network_storage.save_network(network)

    def set_lr(self, lr: float):
        self.current_lr = lr

    def run(self, num_cycles: int, train_first: bool = False):
        for cycle in range(num_cycles):
            if train_first and cycle == 0:
                self.train()
                torch.cuda.empty_cache()
            else:
                self.self_play()
                torch.cuda.empty_cache()
                self.train()
                torch.cuda.empty_cache()

def transform(x):
    return x * 50
