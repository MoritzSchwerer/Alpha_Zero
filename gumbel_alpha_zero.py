import numpy as np
import torch
import math
import copy

from tqdm import tqdm
from typing import List

from network import PredictionNetwork
from game import GameHistory, new_game, Chess
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

    k = int(k)

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


@torch.no_grad
def play_game(config: AlphaZeroConfig, network: PredictionNetwork):
    base_games = [new_game() for _ in range(config.self_play_batch_size)]
    histories = [GameHistory() for _ in range(config.self_play_batch_size)]

    # curr_batch_size = config.self_play_batch_size
    for _ in tqdm(range(config.max_moves)):
        states = [None] * config.self_play_batch_size
        action_masks = [None] * config.self_play_batch_size
        dones = np.zeros(config.self_play_batch_size, dtype=bool)
        for game_idx in range(len(base_games)):
            obs, _, term, trunc, _ = base_games[game_idx].last()

            if term or trunc:
                dones[game_idx] = True
                continue

            states[game_idx] = obs['observation']
            action_masks[game_idx] = obs['action_mask']

        valid_indices = (~dones).nonzero()[0]
        states = [states[idx] for idx in valid_indices]
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
        logits = network(state)[0].cpu().numpy()

        n_logits = np.zeros((config.self_play_batch_size, logits.shape[1]))
        n_logits[valid_indices] = logits

        roots = []
        all_initial_actions = []
        gumbels = []
        all_games = []
        all_histories = []
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
                config,
                root,
                legal_actions,
                gumbel,
                config.num_sampled_actions,
            )
            initial_actions = legal_actions[top_k_ind]

            roots.append(root)
            all_initial_actions.append(initial_actions)
            gumbels.append(gumbel)
            all_games.append(base_games[idx])
            all_histories.append(histories[idx])

        actions = run_sequential_halving(
            config,
            network,
            roots,
            all_initial_actions,
            all_games,
            gumbels,
        )

        assert len(all_games) == len(actions) == len(roots) == len(states)
        for idx, action in enumerate(actions):
            all_games[idx].step(action)

            stats = {}
            for a, c_node in roots[idx].children.items():
                stats[a] = float(
                    c_node.value
                    if c_node.visit_count >= 1
                    else roots[idx].value
                )
            all_histories[idx].store_statistics(
                action,
                root_value=roots[idx].value,
                stats=stats,
                state=states[idx],
            )

    for history, game in zip(histories, base_games):
        history.outcome = game._cumulative_rewards
        history.consolidate()
    return histories


def run_sequential_halving(
    config: AlphaZeroConfig,
    network: PredictionNetwork,
    roots: List[Node],
    actions: List[np.ndarray],
    base_games: List[Chess],
    gumbel: np.ndarray,
) -> List[int]:
    """
    actions shape: list(num_games, np.array(initial_actions))
    gumbel  shape: (num_games, action_space)
    """
    # we get n actions and we want to reduce the number of actions in half for each step
    num_games = len(base_games)

    initial_num_actions = [
        min(len(acs), config.num_sampled_actions) for acs in actions
    ]
    curr_num_actions = initial_num_actions.copy()

    num_performed_sims = np.zeros(num_games)
    num_phases = [
        math.ceil(math.log2(num_actions))
        for num_actions in initial_num_actions
    ]

    for phase in range(max(num_phases)):
        # find if phase is last phase

        max_num_sims = []
        for num_sims, num_actions, num_phase in zip(
            num_performed_sims, curr_num_actions, num_phases
        ):
            is_last_p = phase + 1 == num_phase
            max_num_sims.append(
                int(
                    (config.num_simulations - num_sims) // num_actions
                    if is_last_p
                    else (
                        config.num_simulations
                        // max(num_phase * num_actions, 1)
                    )
                )
            )

        for sim_idx in range(max(max_num_sims)):
            states = np.zeros(
                (num_games, max(curr_num_actions), 8, 8, 111), dtype=bool
            )
            dones = np.zeros((num_games, max(curr_num_actions)), dtype=bool)
            valid_steps = np.zeros(
                (num_games, max(curr_num_actions)), dtype=bool
            )
            action_masks_2d = np.zeros(
                (num_games, max(curr_num_actions), config.action_space_size),
                dtype=bool,
            )
            search_paths_2d = [None] * num_games
            players_2d = [None] * num_games
            outcomes_2d = [None] * num_games
            # for every game collect all actions
            for game_idx in range(num_games):
                if phase >= num_phases[game_idx]:
                    continue
                if sim_idx >= max_num_sims[game_idx]:
                    continue
                search_paths = [None] * curr_num_actions[game_idx]
                players = [None] * curr_num_actions[game_idx]
                outcomes = [None] * curr_num_actions[game_idx]
                for action_idx, r_action in enumerate(actions[game_idx]):
                    if action_idx >= curr_num_actions[game_idx]:
                        break
                    game = copy.deepcopy(base_games[game_idx])
                    game.step(r_action)
                    node = roots[game_idx].children[r_action]
                    search_path = [roots[game_idx], node]
                    while node.is_expanded:
                        # select the action and node according to distribution
                        action, node = select_child(config, node)
                        # step game using selected action
                        game.step(action)
                        # store node for backpropagation
                        search_path.append(node)

                    # run network on leaf node
                    obs, _, term, trunc, _ = game.last()

                    states[game_idx, action_idx] = obs['observation']
                    action_masks_2d[game_idx, action_idx] = obs['action_mask']
                    dones[game_idx, action_idx] = term or trunc
                    valid_steps[game_idx, action_idx] = True

                    search_paths[action_idx] = search_path
                    players[action_idx] = game.agent_selection
                    outcomes[action_idx] = game._cumulative_rewards

                search_paths_2d[game_idx] = search_paths
                players_2d[game_idx] = players
                outcomes_2d[game_idx] = outcomes

            # now we need to ignore the invalid steps
            # and construct only the usefull states
            valid_indices = np.logical_and(
                valid_steps.flatten(), np.logical_not(dones.flatten())
            ).nonzero()[0]

            states = states.reshape(
                (num_games * max(curr_num_actions), 8, 8, 111)
            )

            n_values = np.zeros(states.shape[0])
            n_logits = np.zeros((states.shape[0], config.action_space_size))
            if len(valid_indices) > 0:
                valid_states = states[valid_indices]

                state = (
                    torch.from_numpy(valid_states)
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
                n_logits[valid_indices] = logits

            n_values = n_values.reshape((num_games, max(curr_num_actions)))
            n_logits = n_logits.reshape(
                (num_games, max(curr_num_actions), config.action_space_size)
            )

            for game_idx in range(num_games):
                if search_paths_2d[game_idx] is None:
                    continue
                for action_idx in range(len(search_paths_2d[game_idx])):
                    if dones[game_idx, action_idx]:
                        value = int(
                            outcomes_2d[game_idx][action_idx][
                                players_2d[game_idx][action_idx]
                            ]
                        )
                    else:
                        value = int(n_values[game_idx, action_idx])

                    if (not dones[game_idx, action_idx]) and valid_steps[
                        game_idx, action_idx
                    ]:
                        expand_node(
                            search_paths_2d[game_idx][action_idx][-1],
                            players_2d[game_idx][action_idx],
                            action_masks_2d[game_idx, action_idx],
                            logits=n_logits[game_idx, action_idx],
                        )
                    if valid_steps[game_idx, action_idx]:
                        for s_node in search_paths_2d[game_idx][action_idx]:
                            s_node.visit_count += 1
                            s_node.value_sum += float(
                                value
                                if s_node.cur_player
                                == players_2d[game_idx][action_idx]
                                else -value
                            )
                num_performed_sims[game_idx] += curr_num_actions[game_idx]

        # after all actions have been simulated for the current phase
        # only keep best half of actions
        curr_num_actions = [math.ceil(cna / 2) for cna in curr_num_actions]

        for game_idx in range(num_games):
            top_k_ind = top_k_actions(
                config,
                roots[game_idx],
                actions[game_idx],
                gumbel[game_idx],
                curr_num_actions[game_idx],
            )
            actions[game_idx] = actions[game_idx][top_k_ind]

    return actions
