import copy
from pettingzoo.classic import chess_v6


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


# self.agents = self.possible_agents[:]
#
#         self.board = chess.Board()
#
#         self._agent_selector = AgentSelector(self.agents)
#         self.agent_selection = self._agent_selector.reset()
#
#         self.rewards = {name: 0 for name in self.agents}
#         self._cumulative_rewards = {name: 0 for name in self.agents}
#         self.terminations = {name: False for name in self.agents}
#         self.truncations = {name: False for name in self.agents}
#         self.infos = {name: {} for name in self.agents}
#
#         self.board_history = np.zeros((8, 8, 104), dtype=bool)
