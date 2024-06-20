import numpy as np
import torch
import math
import copy

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


@torch.no_grad
def play_game(config: AlphaZeroConfig, network: PredictionNetwork):
    base_game = new_game()
    history = GameHistory()

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
            for r_action in actions:
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
                state = obs['observation']
                action_mask = obs['action_mask']
                current_player = game.agent_selection

                # NOTE: Maybe here make this batched
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
