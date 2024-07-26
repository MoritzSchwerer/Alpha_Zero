import numpy as np

from typing import Dict


class Node:
    def __init__(self, logit: np.float64, prior: np.float64):
        self.visit_count: int = 0
        self.cur_player: str = ""
        self.logit: np.float64 = logit
        self.prior: np.float64 = prior
        self.value_sum: float = 0
        self.children: Dict[int, Node] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def expand_node(
    node: Node, cur_player: str, action_mask: np.ndarray, logits: np.ndarray
):
    node.cur_player = cur_player
    actions = action_mask.nonzero()[0]
    masked_logits = logits[actions]
    logits = masked_logits - np.max(masked_logits)
    exp_logits = np.exp(logits / 4)
    exp_logits_sum = np.sum(exp_logits)
    priors = exp_logits / exp_logits_sum
    for action, logit, prior in zip(actions, logits, priors):
        node.children[action] = Node(logit, prior)
