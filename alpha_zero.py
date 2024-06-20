import numpy as np
import random
import torch
import math
import copy
import time

from typing import Tuple, List, Any, Optional
from mcts import Node
from network import PredictionNetwork
from pettingzoo.classic import chess_v6
from tqdm import tqdm

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision = 'medium'


class AlphaZeroConfig:
    def __init__(self, new_game_fn):

        self.new_game = new_game_fn
        self.self_play_batch_size = 128
        self.max_num_threads = 8
        self.num_processes = 32

        # gumbel
        self.num_sampled_actions = 4
        self.c_visit = 50
        self.c_scale = 1.0

        # game settings
        self.max_moves = 100
        self.action_space_size = 4672

        # exploration
        self.exploration_fraction = 0.25
        self.dirichlet_alpha = 0.3
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # other
        self.num_simulations = 16

    def softmax_temperature_fn(self, num_moves=30):
        return 1 if num_moves < 30 else 0

    def new_game(self):
        game = chess_v6.env()
        game.reset()
        return game


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


def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.dirichlet_alpha] * len(actions))
    frac = config.exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


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
    visits = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    temperature = config.softmax_temperature_fn(num_moves=num_moves)
    action = softmax_sample(visits, temperature)
    return action


def select_child(config: AlphaZeroConfig, node: Node):

    _, action, child = max(
        (ucb_score(config, node, child), action, child)
        for action, child in node.children.items()
    )
    return action, child


def select_child_gumbel(config: AlphaZeroConfig, node: Node):
    visits = np.array([child.visit_count for child in node.children.values()])
    prior = np.array([child.prior for child in node.children.values()])
    actions = [a for a in node.children.keys()]
    total_visits = np.sum(visits)
    preference = prior - (visits / (total_visits + 1))
    action_index = np.argmax(preference).item()
    action = actions[action_index]
    return action, node.children[action]


def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node) -> float:
    pb_c = (
        math.log(
            (parent.visit_count + config.pb_c_base + 1) / config.pb_c_base
        )
        + config.pb_c_init
    )

    pb_c *= math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    # this transforms it from [-1, 1] to [0, 1]
    value_score = (child.value + 1) / 2
    return prior_score + value_score


def top_k_actions(
    config: AlphaZeroConfig,
    node: Node,
    actions: np.ndarray,
    gumbel: np.ndarray,
    k: int,
) -> np.ndarray:

    if len(actions) <= k:
        return np.array(range(len(actions)))

    max_visit_count = max(
        node.children[action].visit_count for action in actions
    )
    value_func = lambda x: (
        (config.c_visit + max_visit_count) * config.c_scale * x
    )

    gumbel_logits = np.array(
        [
            node.children[action].prior
            + gumbel[i].item()
            + value_func(node.children[action].value)
            for i, action in enumerate(actions)
        ]
    )
    top_k_ind = np.argpartition(gumbel_logits, -k)[-k:]
    return top_k_ind


@torch.no_grad
def play_game_gumbel(config: AlphaZeroConfig, network: PredictionNetwork):

    if torch.cuda.is_available() and torch.backends.cudnn.version() >= 7603:
        network = network.to(
            device=DEVICE, memory_format=torch.channels_last
        ).half()

    base_game = config.new_game()
    history = GameHistory(config.new_game)

    for _ in tqdm(range(config.max_moves)):
        obs, _, term, trunc, _ = base_game.last()

        if term or trunc:
            break

        state = obs['observation']
        state = (
            torch.from_numpy(state.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=DEVICE,
                memory_format=torch.channels_last,
                dtype=torch.float16,
            )
        )
        logits = network(state)[0].flatten().cpu().numpy()

        root = Node(0)
        expand_node(
            root,
            base_game.agent_selection,
            obs['action_mask'],
            logits=logits,
        )

        # after we expanded the root we select the initial n actions
        legal_actions = np.array([a for a in root.children.keys()])
        gumbel = np.random.gumbel(size=len(logits))
        top_k_ind = top_k_actions(
            config, root, legal_actions, gumbel, config.num_sampled_actions
        )
        initial_actions = legal_actions[top_k_ind]

        # don't forget to index gumbel as we already have selected the top n
        # gumbel = gumbel[top_k_ind]

        # step the independent games according to their selected action
        # select final action via sequential halving
        action = run_sequential_halving(
            config, network, root, initial_actions, base_game, gumbel
        )

        assert action in legal_actions
        base_game.step(action)
        history.store_statistics(action)
    history.outcome = base_game._cumulative_rewards
    history.consolidate()
    return history


# TODO: try in run mcts and play_game to store the game inside the node
# to reduce the number of steps needed


def run_sequential_halving(
    config: AlphaZeroConfig,
    network: PredictionNetwork,
    root: Node,
    actions: np.ndarray,
    base_game,
    gumbel: np.ndarray,
) -> int:
    # we get n actions and we want to reduce the number of actions in half for each step
    initial_num_actions = min(len(actions), config.num_sampled_actions)
    curr_num_actions = initial_num_actions

    num_performed_sims = 0
    num_phases = math.ceil(math.log2(initial_num_actions))

    for phase in reversed(range(int(num_phases))):
        # for each action do the correct number of simulations

        if phase == 0:
            sims_left = config.num_simulations - num_performed_sims
            max_num_sims = sims_left // curr_num_actions
        else:
            max_num_sims = int(
                config.num_simulations / (num_phases * curr_num_actions)
            )
        for _ in range(max_num_sims):
            for _, r_action in enumerate(actions):
                game = copy.deepcopy(base_game)
                game.step(r_action)
                node = root.children[r_action]
                search_path = [root, node]
                while node.is_expanded:
                    # select the action and node according to distribution
                    action, node = select_child_gumbel(config, node)
                    # step game using selected action
                    game.step(action)
                    # store node for backpropagation
                    search_path.append(node)

                # if we are done with the while loop, we arrived at a leaf node in the tree

                # run network on leaf node
                obs, _, term, trunc, _ = game.last()
                state = obs['observation']
                action_mask = obs['action_mask']
                current_player = game.agent_selection

                if term or trunc:
                    value = game._cumulative_rewards[current_player]
                else:
                    state = (
                        torch.from_numpy(state.copy())
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(
                            device=DEVICE,
                            memory_format=torch.channels_last,
                            dtype=torch.float16,
                        )
                    )
                    logits, value = network(state)
                    logits, value = (
                        logits.cpu().flatten().numpy(),
                        value.cpu().flatten().numpy(),
                    )
                    # expand node
                    expand_node(
                        search_path[-1],
                        current_player,
                        action_mask,
                        logits=logits,
                    )
                for s_node in search_path:
                    s_node.visit_count += 1
                    s_node.value_sum += float(
                        value
                        if s_node.cur_player == current_player
                        else -value
                    )
                num_performed_sims += 1

        # after all actions have been simulated for the current phase
        # only keep best half of actions
        curr_num_actions = math.ceil(curr_num_actions / 2)

        top_k_ind = top_k_actions(
            config, root, actions, gumbel, curr_num_actions
        )
        actions = actions[top_k_ind]

    assert actions.shape == (1,)
    return actions.item()


def run_mcts(
    config: AlphaZeroConfig,
    network: PredictionNetwork,
    roots: List[Node],
    base_games: List[Any],
):
    net_time = 0
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
            states[idx] = obs['observation']
            action_masks[idx] = obs['action_mask']
            dones[idx] = term or trunc

        n_values = np.zeros(config.self_play_batch_size)
        n_logits = np.zeros(
            (config.self_play_batch_size, config.action_space_size)
        )
        if sum(dones) < config.self_play_batch_size:
            start = time.time_ns()

            valid_indices = (~dones).nonzero()[0]

            state = np.stack(
                [s for i, s in enumerate(states) if i in valid_indices], 0
            )
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

            net_time += time.time_ns() - start
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
        # net_time += time.time_ns() - start

        # backpropagate
        for idx in range(config.self_play_batch_size):
            for node in search_paths[idx]:
                node.visit_count += 1
                node.value_sum += float(
                    n_values[idx]
                    if node.cur_player == cur_player
                    else -n_values[idx]
                )
    print('net time: ', net_time // 1e6, ' ms')


def extract_visit_counts(config: AlphaZeroConfig, root: Node):
    visit_counts = np.zeros(config.action_space_size, dtype=np.int64)
    for action, node in root.children.items():
        visit_counts[action] = node.visit_count
    assert (
        np.sum(visit_counts) == root.visit_count
    ), f'Got {np.sum(visit_counts)} and {root.visit_count}'
    return visit_counts


@torch.no_grad
def play_game(
    config: AlphaZeroConfig, network: PredictionNetwork
) -> List[GameHistory]:
    # game_time = 0
    if torch.cuda.is_available() and torch.backends.cudnn.version() >= 7603:
        network = network.to(
            device=DEVICE, memory_format=torch.channels_last
        ).half()

    # start_game = time.time_ns()
    games = [config.new_game() for _ in range(config.self_play_batch_size)]
    # game_time += time.time_ns() - start_game

    histories = [
        GameHistory(config.new_game)
        for _ in range(config.self_play_batch_size)
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
            states[idx] = obs['observation']
            action_masks[idx] = obs['action_mask']
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
        print('mcts took: ', (time.time_ns() - start) // 1e6, ' seconds')
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
