import numpy as np
import torch
import math
import copy
import time

from typing import Tuple, List, Any
from tqdm import tqdm

from network import PredictionNetworkV2
from game import GameHistory, new_game
from config import AlphaZeroConfig
from mcts import Node, expand_node

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision = "medium"


def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
    frac = config.exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def softmax_sample(dist: List[Tuple[int, int]], temp: float):
    if temp <= 0:
        _, action = max(dist)
        return action

    visit_counts = np.array([toup[0] for toup in dist], dtype=np.int64)
    actions = np.array([toup[1] for toup in dist], dtype=np.int64)
    exp = np.exp(visit_counts / temp)
    probs = exp / np.sum(exp)
    action = np.random.choice(actions, p=probs)
    return action


def select_action(config, num_moves: int, node: Node):
    visits = [(child.visit_count, action) for action, child in node.children.items()]
    temperature = config.softmax_temperature_fn(num_moves=num_moves)
    action = softmax_sample(visits, temperature)
    return action


def select_child(config: AlphaZeroConfig, node: Node):
    _, action, child = max(
        (ucb_score(config, node, child), action, child)
        for action, child in node.children.items()
    )
    return action, child


def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node) -> float:
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )

    pb_c *= math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    # this transforms it from [-1, 1] to [0, 1]
    value_score = (child.value + 1) / 2
    return prior_score + value_score


def extract_visit_counts(config: AlphaZeroConfig, root: Node):
    visit_counts = np.zeros(config.action_space_size, dtype=np.int64)
    for action, node in root.children.items():
        visit_counts[action] = node.visit_count
    assert (
        np.sum(visit_counts) == root.visit_count
    ), f"Got {np.sum(visit_counts)} and {root.visit_count}"
    return visit_counts


def run_mcts(
    config: AlphaZeroConfig,
    network: PredictionNetworkV2,
    roots: List[Node],
    base_games: List[Any],
):
    for _ in range(config.num_simulations):
        games = [copy.deepcopy(game) for game in base_games]
        nodes = [r for r in roots]
        search_paths = [[node] for node in nodes]
        histories: List[List[int]] = [[] for _ in nodes]
        states = [None] * config.self_play_batch_size
        action_masks = [None] * config.self_play_batch_size
        dones = np.zeros(config.self_play_batch_size, dtype=bool)

        for idx in range(config.self_play_batch_size):
            node = nodes[idx]
            while node.is_expanded:
                cur_player = games[idx].agent_selection
                action, node = select_child(config, node)
                action = action.item()
                histories[idx].append(action)
                search_paths[idx].append(node)
                games[idx].step(action)

            obs, _, term, trunc, _ = games[idx].last()
            states[idx] = obs["observation"]
            action_masks[idx] = obs["action_mask"]
            dones[idx] = term or trunc

        n_values = np.zeros(config.self_play_batch_size)
        n_logits = np.zeros((config.self_play_batch_size, config.action_space_size))
        if sum(dones) < config.self_play_batch_size:
            valid_indices = (~dones).nonzero()[0]

            state = np.stack([s for i, s in enumerate(states) if i in valid_indices], 0)
            state = (
                torch.from_numpy(state)
                .permute(0, 3, 1, 2)
                .to(
                    device=DEVICE,
                    memory_format=torch.channels_last,
                    dtype=torch.float16,
                )
            )
            logits, values = network(state)
            logits, values = (
                logits.cpu().numpy(),
                values.cpu().flatten().numpy(),
            )
            n_values[valid_indices] = values
            n_logits[~dones] = logits

        for idx in range(config.self_play_batch_size):
            cur_player = games[idx].agent_selection
            if dones[idx]:
                n_values[idx] = games[idx]._cumulative_rewards[cur_player]
                continue
            expand_node(
                search_paths[idx][-1],
                cur_player=cur_player,
                action_mask=action_masks[idx],
                logits=n_logits[idx],
            )
        # backpropagate
        for idx in range(config.self_play_batch_size):
            for node in search_paths[idx]:
                node.visit_count += 1
                node.value_sum += float(
                    n_values[idx] if node.cur_player == cur_player else -n_values[idx]
                )


@torch.no_grad
def play_game(
    config: AlphaZeroConfig, network: PredictionNetworkV2
) -> List[GameHistory]:
    # game_time = 0
    if torch.cuda.is_available() and torch.backends.cudnn.version() >= 7603:
        network = network.to(device=DEVICE, memory_format=torch.channels_last).half()

    # start_game = time.time_ns()
    games = [new_game() for _ in range(config.self_play_batch_size)]
    # game_time += time.time_ns() - start_game

    histories = [
        GameHistory(config.new_game) for _ in range(config.self_play_batch_size)
    ]
    dones = [False] * config.self_play_batch_size
    states = [None] * config.self_play_batch_size
    action_masks = [None] * config.self_play_batch_size

    # start_game = time.time_ns()
    for _ in tqdm(range(config.max_moves)):
        for idx, game in enumerate(games):
            if dones[idx]:
                continue
            obs, _, term, trunc, _ = game.last()
            dones[idx] = term or trunc
            states[idx] = obs["observation"]
            action_masks[idx] = obs["action_mask"]
        # game_time += time.time_ns() - start_game

        if np.all(dones):
            break

        # start_net = time.time_ns()
        state = np.stack(states, 0)
        state = (
            torch.from_numpy(state)
            .permute(0, 3, 1, 2)
            .to(
                device=DEVICE,
                memory_format=torch.channels_last,
                dtype=torch.float16,
            )
        )

        policies = network(state)[0].cpu().numpy()
        # net_time = time.time_ns() - start_net
        roots = [Node(0) for _ in range(config.self_play_batch_size)]
        for idx in range(config.self_play_batch_size):
            if dones[idx]:
                continue
            expand_node(
                roots[idx],
                games[idx].agent_selection,
                action_masks[idx],
                policies[idx],
            )
            add_exploration_noise(config, roots[idx])

        start = time.time_ns()
        run_mcts(config, network, roots, games)
        print("mcts took: ", (time.time_ns() - start) // 1e6, " seconds")
        for idx in range(config.self_play_batch_size):
            if dones[idx]:
                continue
            action = select_action(config, len(histories[idx]), roots[idx])
            games[idx].step(action)

            visit_counts = extract_visit_counts(config, roots[idx])

            histories[idx].store_statistics(action, visit_counts)
    for idx in range(config.self_play_batch_size):
        histories[idx].outcome = games[idx]._cumulative_rewards
        histories[idx].consolidate()
    return histories
