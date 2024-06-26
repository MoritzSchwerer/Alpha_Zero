from train import Trainer
from config import AlphaZeroConfig
from storage import HDF5Config, ReplayBuffer
from network import NetworkStorage, NetworkConfig

def main():
    game_length = 100

    hdf5_config = HDF5Config(game_length=game_length, compression='lzf')
    replay_buffer = ReplayBuffer(
        hdf5_config,
        base_dir='./replay_buffer/',
        base_name=f'sample_red_chunk_l{game_length}'
    )

    network_config = NetworkConfig(interm_channels=128)
    network_storage = NetworkStorage(network_config, dir='./networks')

    config = AlphaZeroConfig(
        game_length=game_length,
        num_processes=16,
        self_play_batch_size=128
    )
    trainer = Trainer(
        config,
        replay_buffer=replay_buffer,
        network_storage=network_storage
    )

    trainer.self_play()
    print(f"Number of games generated: {trainer.replay_buffer.current_size}")


if __name__ == '__main__':
    main()
