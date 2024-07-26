import numpy as np
import torch
import math
import copy

from typing import List, Any

from network import PredictionNetworkV2, NetworkConfig, categorical_to_float
from game import GameHistory, new_game, Chess
from config import AlphaZeroConfig
from mcts import Node, expand_node

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")


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

    max_visit_count = max(node.children[action].visit_count for action in actions)

    def value_func(x):
        return (config.c_visit + max_visit_count) * config.c_scale * x

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


@torch.no_grad()
def play_game(
    config: AlphaZeroConfig, network_config: NetworkConfig, network: PredictionNetworkV2
):
    base_games = [new_game() for _ in range(config.self_play_batch_size)]
    histories = [GameHistory() for _ in range(config.self_play_batch_size)]
    # network = torch.compile(network, mode="max-autotune")

    # curr_batch_size = config.self_play_batch_size
    for move_idx in range(config.max_moves):
        states = [None] * config.self_play_batch_size
        action_masks = [None] * config.self_play_batch_size
        dones = np.zeros(config.self_play_batch_size, dtype=bool)
        for game_idx in range(len(base_games)):
            obs, _, term, trunc, _ = base_games[game_idx].last()

            if term or trunc:
                dones[game_idx] = True
                continue

            states[game_idx] = obs["observation"]
            action_masks[game_idx] = obs["action_mask"]

        if not np.all(dones):
            valid_indices = (~dones).nonzero()[0]
            states: List[Any] = [states[idx] for idx in valid_indices]
            state = np.stack(states, 0)
            state = (
                torch.from_numpy(state)
                .permute(0, 3, 1, 2)
                .to(
                    device=network_config.device,
                    memory_format=torch.channels_last
                    if network_config.channels_last
                    else torch.contiguous_format,
                    dtype=torch.float16 if network_config.half else torch.float32,
                )
            )
            logits, values = network(state)
            if torch.any(torch.isnan(logits)) or torch.any(torch.isnan(values)):
                raise ValueError("Got nans in network output.")
            values = categorical_to_float(values)
            logits = logits.cpu().numpy()
            values = values.cpu().flatten().numpy()

            n_logits = np.zeros(
                (config.self_play_batch_size, logits.shape[1]), dtype=np.float32
            )
            n_logits[valid_indices] = logits
            n_values = np.zeros(config.self_play_batch_size, dtype=np.float32)
            n_values[valid_indices] = values

            roots = []
            all_initial_actions = []
            gumbels = []
            all_games = []
            all_histories = []
            for idx in range(len(base_games)):
                if dones[idx]:
                    continue

                root = Node(0, 0)
                expand_node(
                    root,
                    base_games[idx].agent_selection,
                    action_masks[idx],
                    logits=n_logits[idx],
                )
                root.visit_count += 1
                root.value_sum += float(n_values[idx])

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
                network_config,
                network,
                roots,
                all_initial_actions,
                all_games,
                gumbels,
            )

            # sometimes there is a one element array in this list which crashes the program
            actions = [int(a) for a in actions]

            assert len(all_games) == len(actions) == len(roots) == len(states)

            for idx, action in enumerate(actions):
                all_games[idx].step(action)

                stats = {}
                if roots[idx].visit_count > 1:
                    value = _calc_mix_value(roots[idx])

                    for a, c_node in roots[idx].children.items():
                        stats[a] = float(
                            c_node.value if c_node.visit_count >= 1 else value
                        )
                else:
                    stats[action] = roots[idx].value

                all_histories[idx].store_statistics(
                    action,
                    root_value=float(value),
                    stats=stats,
                    state=states[idx],
                )

    for history, game in zip(histories, base_games):
        history.outcome = game._cumulative_rewards
        history.consolidate()
    torch.cuda.empty_cache()
    return histories


def run_sequential_halving(
    config: AlphaZeroConfig,
    network_config: NetworkConfig,
    network: PredictionNetworkV2,
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

    initial_num_actions = [min(len(acs), config.num_sampled_actions) for acs in actions]
    curr_num_actions = initial_num_actions.copy()

    num_performed_sims = np.zeros(num_games)
    num_phases = [
        math.ceil(math.log2(num_actions)) for num_actions in initial_num_actions
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
                    else (config.num_simulations // max(num_phase * num_actions, 1))
                )
            )

        for sim_idx in range(max(max_num_sims)):
            states = np.zeros((num_games, max(curr_num_actions), 8, 8, 111), dtype=bool)
            dones = np.zeros((num_games, max(curr_num_actions)), dtype=bool)
            valid_steps = np.zeros((num_games, max(curr_num_actions)), dtype=bool)
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
                        print("I think this should never get hit")
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

                    states[game_idx, action_idx] = obs["observation"]
                    action_masks_2d[game_idx, action_idx] = obs["action_mask"]
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

            states = states.reshape((num_games * max(curr_num_actions), 8, 8, 111))

            n_values = np.zeros(states.shape[0])
            n_logits = np.zeros((states.shape[0], config.action_space_size))
            if len(valid_indices) > 0:
                valid_states = states[valid_indices]

                state = (
                    torch.from_numpy(valid_states)
                    .permute(0, 3, 1, 2)
                    .to(
                        device=network_config.device,
                        memory_format=torch.channels_last
                        if network_config.channels_last
                        else torch.contiguous_format,
                        dtype=torch.float16 if network_config.half else torch.float32,
                    )
                )
                logits, values = network(state)
                values = categorical_to_float(values)
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
                        value = float(
                            outcomes_2d[game_idx][action_idx][
                                players_2d[game_idx][action_idx]
                            ]
                        )
                    else:
                        value = float(n_values[game_idx, action_idx])

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
                                if s_node.cur_player == players_2d[game_idx][action_idx]
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


def _calc_mix_value(root: Node):
    """
    calculate mix value here see
    https://openreview.net/forum?id=bERaNdoegnO eq. 33
    """
    chosen_action_stats = []
    for action, child in root.children.items():
        if child.visit_count > 0:
            chosen_action_stats.append((action, child))

    # simplified algorithm
    value = 0.0
    logit_sum = 0.0
    for _, c in chosen_action_stats:
        value += c.value * c.logit
        logit_sum += c.logit
    value /= logit_sum + 1e-5
    value += root.value / root.visit_count

    # weighted_priors = sum(c.visit_count * c.prior for _, c in chosen_action_stats)
    # weight = root.visit_count / weighted_priors
    # value_mix = sum(c.prior * c.value for _, c in chosen_action_stats)
    # unscaled_value_approx = root.value + weight * value_mix
    # scale = 1 / (root.visit_count + 1)
    # value = scale * unscaled_value_approx
    return value
