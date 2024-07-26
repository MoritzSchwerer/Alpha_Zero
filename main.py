from train import Trainer
from config import AlphaZeroConfig
from storage import HDF5Config, ReplayBuffer
from network import NetworkStorage, NetworkConfig


def main():
    game_length = 500

    hdf5_config = HDF5Config(game_length=game_length)
    replay_buffer = ReplayBuffer(
        hdf5_config,
        base_dir="./replay_buffer/",
        # base_dir='/tmp/replay_buffer/',
        base_name=f"second_run_l{game_length}",
    )

    network_config = NetworkConfig(
        interm_channels=64,
        bottleneck_channels=32,
        num_blocks=24,
        half=False,
    )
    network_storage = NetworkStorage(
        network_config,
        dir="./networks",
        # dir='/tmp/networks/',
        prefix=f"second_run_l{game_length}",
    )

    config = AlphaZeroConfig(
        game_length=game_length,
        num_processes=40,
        max_num_threads=8,
        self_play_batch_size=64,
    )
    trainer = Trainer(
        config,
        replay_buffer=replay_buffer,
        network_storage=network_storage,
        beta_value=1.0,
        lr=0.2,
        batch_size=4096,
    )

    trainer.run(num_cycles=20, train_first=False)
    print(f"Number of games generated: {trainer.replay_buffer.current_size}")


if __name__ == "__main__":
    main()
