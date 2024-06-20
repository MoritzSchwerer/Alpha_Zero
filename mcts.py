import numpy as np

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


def expand_node(
    node: Node, cur_player: str, action_mask: np.ndarray, logits: np.ndarray
):
    node.cur_player = cur_player
    actions = action_mask.nonzero()[0]
    masked_logits = logits[actions]
    exp_logits = np.exp(masked_logits - np.max(masked_logits))
    policy = exp_logits / np.sum(exp_logits, axis=0)
    for action, p in zip(actions, policy):
        node.children[action] = Node(p)
