from train import Trainer
from config import AlphaZeroConfig
from storage import HDF5Config, ReplayBuffer
from network import NetworkStorage, NetworkConfig

def main():
    game_length = 500

    hdf5_config = HDF5Config(game_length=game_length)
    replay_buffer = ReplayBuffer(
        hdf5_config,
        base_dir='./replay_buffer/',
        # base_dir='/tmp/replay_buffer/',
        base_name=f'second_run_l{game_length}'
    )

    network_config = NetworkConfig(interm_channels=128, num_blocks=19, use_bn=True)
    network_storage = NetworkStorage(
        network_config,
        dir='./networks',
        # dir='/tmp/networks/',
        prefix=f'second_run_l{game_length}'
    )

    config = AlphaZeroConfig(
        game_length=game_length,
        num_processes=40,
        max_num_threads=8,
        self_play_batch_size=128,
    )
    trainer = Trainer(
        config,
        replay_buffer=replay_buffer,
        network_storage=network_storage,
        lr=0.2,
    )

    trainer.run(num_cycles=2, train_first=False)
    print(f"Number of games generated: {trainer.replay_buffer.current_size}")


if __name__ == '__main__':
    main()
