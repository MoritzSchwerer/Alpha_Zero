import copy
import numpy as np

from pettingzoo.classic import chess_v6
from typing import Optional, Tuple


class Chess(chess_v6.raw_env):
    def __deepcopy__(self, memo):
        clone = Chess()
        clone.agents = copy.deepcopy(self.agents)
        clone.board = copy.deepcopy(self.board)
        clone.agent_selection = copy.deepcopy(self.agent_selection)
        clone._agent_selector = copy.deepcopy(self._agent_selector)
        clone.rewards = copy.deepcopy(self.rewards)
        clone._cumulative_rewards = copy.deepcopy(self._cumulative_rewards)
        clone.terminations = copy.deepcopy(self.terminations)
        clone.truncations = copy.deepcopy(self.truncations)
        clone.infos = copy.deepcopy(self.infos)
        clone.board_history = self.board_history.copy()
        return clone


class GameHistory:
    def __init__(self, create_func):
        self.new_game = create_func
        self.actions = []
        self.search_stats = []
        self.outcome = None

    def store_statistics(
        self, action: int, stats: Optional[np.ndarray] = None
    ) -> None:
        self.actions.append(action)
        if stats is not None:
            self.search_stats.append(stats)

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def consolidated(self) -> bool:
        return isinstance(self.actions, np.ndarray)

    def consolidate(self):
        """
        enables object to be stored more efficiently
        """
        self.actions = np.array(self.actions, dtype=np.int64)
        if len(self.search_stats) > 0:
            self.search_stats = np.stack(self.search_stats, 0)

    def sample_position(self) -> int:
        assert self.consolidated, 'Object needs to be consolidated first'
        move_index = random.randint(0, len(self) - 1)
        return move_index

    def get_observation(self, move_index: int) -> dict[str, np.ndarray]:
        game = self.new_game()

        for i in range(move_index):
            game.step(self.actions[i])

        obs = game.last()[0]
        return obs

    def cur_player(self, move_index: int) -> str:
        if move_index % 2 == 0:
            return 'player_0'
        return 'player_1'

    def get_targets(self, move_index: int) -> Tuple[np.ndarray, int]:
        return (
            self.search_stats[move_index],
            self.outcome[self.cur_player(move_index)],
        )
