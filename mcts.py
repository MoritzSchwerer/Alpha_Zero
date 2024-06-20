from typing import Dict


class Node:
    def __init__(self, prior: float):
        self.visit_count: int = 0
        self.cur_player: str = ''
        self.prior: float = prior
        self.value_sum: float = 0
        self.children: Dict[int, Node] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self):
        pass
