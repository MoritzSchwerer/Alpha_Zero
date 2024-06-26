import numpy as np
import h5py
import random
import os
import time
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
    ):
        self.base_dir = base_dir
        self.base_name = base_name
        self.config = config
        self.current_files: Deque[str] = deque(maxlen=self.config.num_files)
        self.current_size = 0
        self.created = False
        self.new_file_name = FileNamer(base=self.base_name)
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

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
            compression=self.config.compression
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
                compression=self.config.compression
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

    def sample_examples_fast(self, batch_size=4096):
        assert self.created, "Must create_dataset or init_from_dir first"

        num_files = 4
        chunk_size = self.config.chunk_size
        elems_per_file = batch_size // num_files

        # pick n files
        selected_files = random.choices(self.current_files, k=num_files)
        num_chunks_per_file: Dict[str, int] = {f: 0 for f in selected_files}
        for f in selected_files:
            num_chunks_per_file[f] += elems_per_file // chunk_size

        states_list = []
        outcomes_list = []
        avs_list = []
        for file, num_chunks in num_chunks_per_file.items():
            total_num_chunks = self.config.file_size // chunk_size
            selected_chunks = np.random.choice(
                total_num_chunks, size=num_chunks, replace=False
            )
            random_time = 0
            states_time = 0
            av_time = 0
            outcomes_time = 0
            with h5py.File(file, "r") as f:
                print(f['states'].chunks)
                print(len(list(f['states'].iter_chunks())))
                for chunk in selected_chunks:
                    start = chunk * chunk_size
                    end = start + chunk_size

                    start_t = time.time_ns()
                    lengths = np.array(f["lengths"][start:end])
                    move_indices = np.zeros(len(lengths), dtype=np.int32)
                    for i, length in enumerate(lengths):
                        move_indices[i] = random.randint(0, length - 1)
                    random_time += (time.time_ns()-start_t)

                    start_t = time.time_ns()
                    states = f['states'][start:end]
                    states = states[np.arange(chunk_size), move_indices]

                    # print(f"single time: {int((time.time_ns()-start_t)/1e6)} ms.")
                    states_time += (time.time_ns()-start_t)

                    start_t = time.time_ns()
                    avs = f["action_values"][start:end]
                    avs = [avs[i][idx] for i, idx in enumerate(move_indices)]
                    av_list = []
                    for av in avs:
                        # TODO: make 4672 a parameter
                        arr = np.zeros(4672, dtype=np.float32)
                        for a, v in av:
                            arr[a] = v
                        av_list.append(arr)
                    avs = np.stack(av_list, 0)
                    av_time += (time.time_ns()-start_t)

                    start_t = time.time_ns()
                    players = lengths % 2
                    outcomes = f["outcomes"][start:end]
                    outcomes = np.array([outcomes[i][p] for i, p in enumerate(players)], dtype=np.int64)
                    outcomes_time += (time.time_ns()-start_t)

                    states_list.append(states)
                    avs_list.append(avs)
                    outcomes_list.append(outcomes)

        print("="*88)
        print(f"Random took: {int(random_time/1e6)} ms")
        print(f"States took: {int(states_time/1e6)} ms")
        print(f"AV     took: {int(av_time/1e6)} ms")
        print(f"outcom took: {int(outcomes_time/1e6)} ms")
        states = np.concatenate(states_list, 0).reshape(-1, 111, 8, 8)
        outcomes = np.concatenate(outcomes_list, 0)
        action_values = np.concatenate(avs_list, 0)
        return states, outcomes, action_values

    # TODO: the slice approach is probably the best
    # the idea is to get all the lengths
    # then randomly select chunks and random points in the respective
    # game, construct the slices and then index into the dataset
    # via the slices
    # this way we load all the length values and
    # only load the neccessary other values
    # playing with chunk size probably makes sence then
    def sample_examples(self, batch_size=4096):
        assert self.created, "Must create_dataset or init_from_dir first"

        num_files = 4
        chunk_size = self.config.chunk_size
        elems_per_file = batch_size // num_files

        # pick n files
        selected_files = random.choices(self.current_files, k=num_files)
        num_elems_per_file: Dict[str, int] = {f: 0 for f in selected_files}
        for f in selected_files:
            num_elems_per_file[f] += elems_per_file

        states_list = []
        outcomes_list = []
        avs_list = []
        for file, num_elems in num_elems_per_file.items():
            # random_time = 0
            # states_time = 0
            # av_time = 0
            # outcomes_time = 0
            with h5py.File(file, "r") as f:
                # print(len(list(f['states'].iter_chunks())))
                chunk_list = list(f['states'].iter_chunks())
                # print(type(chunk_list))
                selected_chunks = random.sample(chunk_list, k=(batch_size)//chunk_size)
                for chunk in selected_chunks:
                    states = f['states'][chunk]
                    avs = f['action_values'][chunk[0]]
                    outcomes = f['outcomes'][chunk[0]]
                    av_list = []
                    # for av in avs:
                    #     # TODO: make 4672 a parameter
                    #     arr = np.zeros(4672, dtype=np.float32)
                    #     for a, v in av:
                    #         arr[a] = v
                    #     av_list.append(arr)
                    # avs = np.stack(av_list, 0)
                states_list.append(states)
                avs_list.append(avs)
                outcomes_list.append(outcomes)

        # print("="*88)
        # print(f"Random took: {int(random_time/1e6)} ms")
        # print(f"States took: {int(states_time/1e6)} ms")
        # print(f"AV     took: {int(av_time/1e6)} ms")
        # print(f"outcom took: {int(outcomes_time/1e6)} ms")
        states = np.concatenate(states_list, 0).reshape(-1, 111, 8, 8)
        action_values = None
        # outcomes = np.concatenate(outcomes_list, 0)
        # action_values = np.concatenate(avs_list, 0)
        return states, outcomes, action_values


def _read_lengths(file_name: str):
    with h5py.File(file_name, "r") as f:
        lengths = np.array(f["lengths"], dtype=np.int32)
    return lengths


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
            f["action_values"][start_idx + i] = np.array(
                ls, dtype=h5py.special_dtype(vlen=dict_dt)
            )


def _create_dataset(
    file_name: str = "/tmp/a.h5",
    size: int = 1024,
    game_length: int = 100,
    chunk_size: int = 256,
    compression: Optional['str'] = None,
):
    with h5py.File(file_name, "w") as f:
        dict_dt = np.dtype([("key", np.int32), ("value", np.float32)])
        outcome_dt = np.dtype([("player_0", np.int32), ("player_1", np.int32)])
        f.create_dataset(
            "actions",
            (size, game_length),
            dtype=np.int16,
            # compression=compression,
            chunks=(chunk_size, game_length),
        )
        f.create_dataset(
            "lengths",
            (size,),
            dtype=np.int16,
            # compression=compression,
            chunks=(chunk_size,),
        )
        f.create_dataset(
            "root_values",
            (size, game_length),
            dtype=np.float16,
            compression=compression,
            chunks=(chunk_size, game_length),
        )
        f.create_dataset(
            "action_values",
            (size,),
            dtype=h5py.special_dtype(vlen=h5py.special_dtype(vlen=dict_dt)),
            compression=compression,
            chunks=(chunk_size,),
        )
        f.create_dataset(
            "states",
            (size, game_length, 111 * 8 * 8),
            dtype=np.bool_,
            compression=compression,
            chunks=(chunk_size, game_length, 111*8*8),
        )
        f.create_dataset(
            "outcomes",
            (size,),
            dtype=outcome_dt,
            # compression=compression,
            chunks=(chunk_size,),
        )


class FileNamer:
    def __init__(self, base: str, fill_length: int = 4, ext: str = ".h5"):
        self.base = base
        self.count = 1
        self.fill_length = fill_length
        self.ext = ext

    def __call__(self):
        name = self.base + f"_{str(self.count).zfill(self.fill_length)}{self.ext}"
        self.count += 1
        return name
