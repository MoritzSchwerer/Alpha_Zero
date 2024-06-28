import numpy as np
import h5py
import random
import os
import hdf5plugin
from pathlib import Path

from game import GameHistory
from typing import List, Deque, Dict, Optional
from collections import deque


class HDF5Config:
    def __init__(
        self,
        max_size: int = 1024 * 1000,
        file_size: int = 51200,
        chunk_size: int = 256,
        game_length: int = 100,
        compression: Optional[str] = None,
    ):
        self.max_size = max_size
        self.file_size = file_size
        self.chunk_size = chunk_size
        self.game_length = game_length
        self.num_files = self.max_size // self.file_size
        self.compression = compression


class ReplayBuffer:
    def __init__(
        self,
        config: HDF5Config,
        base_dir: str = "./replay_buffer/",
        base_name="rb",
        num_sample_files: int = 4,
        total_num_actions: int = 4672,
    ):
        self.base_dir = base_dir
        self.base_name = base_name
        self.config = config
        self.num_sample_files = num_sample_files
        self.total_num_actions = total_num_actions
        self.current_files: Deque[str] = deque(maxlen=self.config.num_files)
        self.current_size = 0
        self.created = False

        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        num_files = len(list(Path(self.base_dir).glob(f"{self.base_name}*.h5")))
        self.new_file_name = FileNamer(base=self.base_name, start_count=num_files+1)

    def init_from_dataset(self) -> bool:
        """
        returns True if succefully initialized with existing dataset
        if no files with the correct pattern could be found returns False
        """
        root_dir = Path(self.base_dir)
        files = list(
            sorted(
                map(str, root_dir.glob(f"{self.base_name}*{self.new_file_name.ext}"))
            )
        )
        if len(files) > 0:
            self.current_files.extend(files)
            self.current_size = (len(files) - 1) * self.config.file_size
            lengths = _read_lengths(self.current_files[-1])
            assert lengths.shape[0] == self.config.file_size
            non_zero = lengths.nonzero()[0]
            if len(non_zero) == 0:
                num_elements = 0
            else:
                num_elements = non_zero[-1] + 1
            self.current_size = (len(files) - 1) * self.config.file_size + num_elements
            self.created = True
            return True
        return False

    def create_dataset(self):
        file_name = self.base_dir + self.new_file_name()
        _create_dataset(
            file_name=file_name,
            size=self.config.file_size,
            game_length=self.config.game_length,
            chunk_size=self.config.chunk_size,
            compression=self.config.compression,
        )
        self.current_files.append(file_name)
        self.created = True

    def append(self, data):
        assert self.created, "Must create_dataset or init_from_dir first"
        if self.current_size % self.config.file_size == 0 and self.current_size > 0:
            file_name = self.base_dir + self.new_file_name()
            _create_dataset(
                file_name=file_name,
                size=self.config.file_size,
                game_length=self.config.game_length,
                compression=self.config.compression,
            )
            if len(self.current_files) == self.config.num_files:
                self.current_size -= self.config.file_size
            self.current_files.append(file_name)

        _append_dataset(
            data,
            self.current_size % self.config.file_size,
            file_name=self.current_files[-1],
            game_length=self.config.game_length,
        )
        self.current_size += len(data)


    def sample_examples(self, batch_size=4096):
        assert self.created, "Must create_dataset or init_from_dir first"

        selected_files = random.choices(self.current_files, k=self.num_sample_files)
        num_elems_per_file: Dict[str, int] = {f: 0 for f in selected_files}
        for f in selected_files:
            num_elems_per_file[f] += batch_size // self.num_sample_files

        states_list = []
        outcomes_list = []
        action_values_list = []
        for file, num_elems in num_elems_per_file.items():
            # NOTE: this might need fixing
            example_games = np.random.choice(
                self.config.file_size, size=num_elems, replace=False
            )
            example_games = np.sort(example_games)
            states, outcomes, action_values = _sample_single(file, example_games, self.total_num_actions)
            states_list.append(states)
            outcomes_list.append(outcomes)
            action_values_list.append(action_values)

        states = np.concatenate(states_list, 0)
        outcomes = np.concatenate(outcomes_list, 0)
        action_values = np.concatenate(action_values_list, 0)

        return states, outcomes, action_values


def _read_lengths(file_name: str):
    with h5py.File(file_name, "r") as f:
        lengths = np.array(f["lengths"], dtype=np.int32)
    return lengths


def _sample_single(file: str, example_games, total_num_actions=4672):
    states = np.zeros((len(example_games), 7104), dtype=np.bool_)
    lengths = np.zeros((len(example_games)), dtype=np.int32)
    with h5py.File(file, "r") as f:
        # start = time.time_ns()
        for dest_idx, game_idx in enumerate(example_games):
            f["lengths"].read_direct(
                lengths,
                source_sel=np.s_[game_idx],
                dest_sel=np.s_[dest_idx],
            )
        # print("="*88)
        # print(f"Length takes: {int((time.time_ns()-start)/1e6)} ms")
        # start = time.time_ns()
        move_indices = np.zeros(len(lengths), dtype=np.int32)
        for i, length in enumerate(lengths):
            move_indices[i] = np.random.randint(0, length, size=None)
        # print(f"Moves takes: {int((time.time_ns()-start)/1e6)} ms")

        # start = time.time_ns()
        for i, (game_idx, move_idx) in enumerate(zip(example_games, move_indices)):
            f["states"].read_direct(
                states,
                source_sel=np.s_[game_idx, move_idx, :],
                dest_sel=np.s_[i, :],
            )
        # print(f"State takes: {int((time.time_ns()-start)/1e6)} ms")

        root_values = np.zeros(len(example_games), dtype=np.float32)
        for dest_idx, (game_idx, move_idx) in enumerate(zip(example_games, move_indices)):
            f["root_values"].read_direct(
                root_values,
                source_sel=np.s_[game_idx, move_idx],
                dest_sel=np.s_[dest_idx],
            )

        # start = time.time_ns()
        avs = [
            f["action_values"][game_idx, move_idx]
            for game_idx, move_idx in zip(example_games, move_indices)
        ]
        # print(f"Action-Values access takes: {int((time.time_ns()-start)/1e6)} ms")
        # start = time.time_ns()
        # avs_arr = np.full((len(avs), 4672), root_values[], dtype=np.float32)
        action_values = np.broadcast_to(root_values.reshape(-1, 1), (len(root_values), total_num_actions)).copy()
        for i, av in enumerate(avs):
            for a, v in av:
                action_values[i, a] = v
        # print(f"Action-Values policy construction takes: {int((time.time_ns()-start)/1e6)} ms")

        # start = time.time_ns()
        players = lengths % 2
        outcomes = [f["outcomes"][game_idx] for game_idx in example_games]
        outcomes = np.array(
            [outcomes[i][p] for i, p in enumerate(players)], dtype=np.float32
        )
        # print(f"Outcomes takes: {int((time.time_ns()-start)/1e6)} ms")

    return states, outcomes, action_values


def _sample_indices(num_elements, num_samples=100):
    """
    start is included
    end is not included
    """
    # return np.sort(np.random.randint(start, end, size=num_samples))
    return np.sort(np.random.choice(num_elements, size=num_samples, replace=False))


def _preprocess_data(data: List[GameHistory], game_length: int = 100):
    actions_list = []
    states_list = []
    root_values_list = []
    lengths = []
    for game in data:
        curr_length = len(game.actions)
        actions = np.pad(
            game.actions, (0, game_length - curr_length), mode="constant"
        ).astype(np.int16)
        states = game.states
        states = np.pad(
            game.states,
            ((0, game_length - curr_length), (0, 0), (0, 0), (0, 0)),
            mode="constant",
        )
        root_values = np.pad(
            game.root_values, (0, game_length - curr_length), mode="constant"
        ).astype(np.float16)
        states_list.append(states.reshape(game_length, 111 * 8 * 8))
        actions_list.append(actions)
        root_values_list.append(root_values)
        lengths.append(len(game.actions))
    lengths = np.array(lengths, dtype=np.int16)
    return actions_list, states_list, root_values_list, lengths


def _append_dataset(
    data: List[GameHistory],
    start_idx: int,
    game_length: int = 100,
    file_name: str = "/tmp/a.h5",
):
    data_length = len(data)
    actions, states, root_values, lengths = _preprocess_data(data, game_length)

    dict_dt = np.dtype([("key", np.int32), ("value", np.float32)])
    with h5py.File(file_name, "a") as f:
        f["actions"][start_idx : start_idx + data_length] = actions
        f["states"][start_idx : start_idx + data_length] = states
        f["root_values"][start_idx : start_idx + data_length] = root_values
        f["lengths"][start_idx : start_idx + data_length :] = lengths

        for i, game in enumerate(data):
            f["outcomes"][start_idx + i] = tuple(game.outcome.values())
        for i, game in enumerate(data):
            ls = []
            for dct in game.search_stats:
                ls.append(
                    np.array(
                        [(key, value) for key, value in dct.items()],
                        dtype=dict_dt,
                    )
                )
            for _ in range(game_length - len(game.search_stats)):
                ls.append(np.array([], dtype=dict_dt))
            f["action_values"][start_idx + i] = np.array(
                ls, dtype=h5py.special_dtype(vlen=dict_dt)
            )


def _create_dataset(
    file_name: str = "/tmp/a.h5",
    size: int = 1024,
    game_length: int = 100,
    chunk_size: int = 256,
    compression: Optional["str"] = None,
):
    with h5py.File(file_name, "w") as f:
        dict_dt = np.dtype([("key", np.int32), ("value", np.float32)])
        outcome_dt = np.dtype([("player_0", np.int32), ("player_1", np.int32)])
        f.create_dataset(
            "actions",
            (size, game_length),
            dtype=np.int16,
            chunks=(1, 1),
        )
        f.create_dataset(
            "lengths",
            (size,),
            dtype=np.int16,
            chunks=(1,),
        )
        f.create_dataset(
            "root_values",
            (size, game_length),
            dtype=np.float16,
            chunks=(1, 1),
        )
        f.create_dataset(
            "action_values",
            (size, game_length),
            dtype=h5py.special_dtype(vlen=dict_dt),
            **hdf5plugin.Zstd(clevel=1),
            chunks=(1, 1),
        )
        f.create_dataset(
            "states",
            (size, game_length, 111 * 8 * 8),
            dtype=np.bool_,
            **hdf5plugin.Zstd(clevel=1),
            chunks=(1, 1, 111 * 8 * 8),
        )
        f.create_dataset(
            "outcomes",
            (size,),
            dtype=outcome_dt,
            chunks=(1,),
        )


class FileNamer:
    def __init__(self, base: str, fill_length: int = 4, ext: str = ".h5", start_count: int = 1):
        self.base = base
        self.count = start_count
        self.fill_length = fill_length
        self.ext = ext

    def __call__(self):
        name = self.base + f"_{str(self.count).zfill(self.fill_length)}{self.ext}"
        self.count += 1
        return name
