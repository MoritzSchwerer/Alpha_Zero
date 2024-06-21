import numpy as np
import torch
import math
import copy
import sys

from tqdm import tqdm

from network import PredictionNetwork
from game import GameHistory, new_game
from config import AlphaZeroConfig
from mcts import Node, expand_node

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision = 'medium'


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


def select_child(config: AlphaZeroConfig, node: Node):
    visits = np.array([child.visit_count for child in node.children.values()])
    prior = np.array([child.prior for child in node.children.values()])
    actions = [a for a in node.children.keys()]
    total_visits = np.sum(visits)
    preference = prior - (visits / (total_visits + 1))
    action_index = np.argmax(preference).item()
    action = actions[action_index]
    return action, node.children[action]


# TODO: batch the play_game function, this should be something like a 30% speed improvement
@torch.no_grad
def play_game(config: AlphaZeroConfig, network: PredictionNetwork):
    base_games = [new_game() for _ in range(config.self_play_batch_size)]
    histories = [GameHistory() for _ in range(config.self_play_batch_size)]

    # curr_batch_size = config.self_play_batch_size
    for _ in tqdm(range(config.max_moves)):
        states = [None] * config.self_play_batch_size
        action_masks = [None] * config.self_play_batch_size
        dones = np.zeros(config.self_play_batch_size, dtype=bool)
        for idx in range(len(base_games)):
            obs, _, term, trunc, _ = base_games[idx].last()

            if term or trunc:
                dones[idx] = True
                continue

            states[idx] = obs['observation']
            action_masks[idx] = obs['action_mask']

        valid_indices = (~dones).nonzero()[0]
        state = [states[idx] for idx in valid_indices]
        state = np.stack(state, 0)
        state = (
            torch.from_numpy(state)
            .permute(0, 3, 1, 2)
            .to(
                device=DEVICE,
                memory_format=torch.channels_last,
                dtype=torch.float16,
            )
        )
        logits = network(state)[0].cpu().numpy()

        n_logits = np.zeros((config.self_play_batch_size, logits.shape[1]))
        n_logits[valid_indices] = logits

        for idx in range(len(base_games)):

            if dones[idx]:
                continue

            root = Node(0)
            expand_node(
                root,
                base_games[idx].agent_selection,
                action_masks[idx],
                logits=n_logits[idx],
            )

            # after we expanded the root we select the initial n actions
            legal_actions = np.array([a for a in root.children.keys()])
            gumbel = np.random.gumbel(size=config.action_space_size)
            top_k_ind = top_k_actions(
                config, root, legal_actions, gumbel, config.num_sampled_actions
            )
            initial_actions = legal_actions[top_k_ind]

            # step the independent games according to their selected action
            # select final action via sequential halving
            action = run_sequential_halving(
                config, network, root, initial_actions, base_games[idx], gumbel
            )

            assert action in legal_actions
            base_games[idx].step(action)
            histories[idx].store_statistics(action)

    for history, game in zip(histories, base_games):
        history.outcome = game._cumulative_rewards
        history.consolidate()
    return histories


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

        max_num_sims = (
            (config.num_simulations - num_performed_sims) // curr_num_actions
            if phase == 0
            else (config.num_simulations // (num_phases * curr_num_actions))
        )

        for _ in range(max_num_sims):
            # TODO: here is where you loop over all games
            states = []
            action_masks = []
            search_paths = []
            players = []
            dones = []
            outcomes = []
            for idx, r_action in enumerate(actions):
                game = copy.deepcopy(base_game)
                game.step(r_action)
                node = root.children[r_action]
                search_path = [root, node]
                while node.is_expanded:
                    # select the action and node according to distribution
                    action, node = select_child(config, node)
                    # step game using selected action
                    game.step(action)
                    # store node for backpropagation
                    search_path.append(node)

                # run network on leaf node
                obs, _, term, trunc, _ = game.last()
                current_player = game.agent_selection
                states.append(obs['observation'])
                action_masks.append(obs['action_mask'])
                search_paths.append(search_path)
                players.append(current_player)
                dones.append(term or trunc)
                outcomes.append(game._cumulative_rewards)

            state = (
                torch.from_numpy(np.stack(states, 0))
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
            n_values = [
                outcomes[idx][players[idx]] if dones[idx] else values[idx]
                for idx in range(len(outcomes))
            ]

            for idx in range(len(search_paths)):
                if not dones[idx]:
                    expand_node(
                        search_paths[idx][-1],
                        players[idx],
                        action_masks[idx],
                        logits=logits[idx],
                    )
                for s_node in search_paths[idx]:
                    s_node.visit_count += 1
                    s_node.value_sum += float(
                        n_values[idx]
                        if s_node.cur_player == current_player
                        else -n_values[idx]
                    )
            num_performed_sims += curr_num_actions

        # after all actions have been simulated for the current phase
        # only keep best half of actions
        curr_num_actions = math.ceil(curr_num_actions / 2)

        top_k_ind = top_k_actions(
            config, root, actions, gumbel, curr_num_actions
        )
        actions = actions[top_k_ind]

    assert actions.shape == (1,)
    return actions.item()
