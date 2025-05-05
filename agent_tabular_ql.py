"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Selects an (action, object) using epsilon-greedy policy."""
    
    if np.random.rand() < epsilon:
        # Explore: choose random action and object
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)
    else:
        # Exploit: choose the action-object pair with highest Q-value
        q_values = q_func[state_1, state_2, :, :]  # shape (NUM_ACTIONS, NUM_OBJECTS)
        action_index, object_index = np.unravel_index(np.argmax(q_values), q_values.shape)

    return (action_index, object_index)


# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition"""
    
    # Get current Q-value
    current_q = q_func[current_state_1, current_state_2, action_index, object_index]
    
    if terminal:
        target = reward
    else:
        # Compute max over all actions at the next state
        next_q_values = q_func[next_state_1, next_state_2, :, :]
        target = reward + GAMMA * np.max(next_q_values)

    # Q-learning update
    updated_q = (1 - ALPHA) * current_q + ALPHA * target
    q_func[current_state_1, current_state_2, action_index, object_index] = updated_q

    return None

# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """Runs one episode of the game"""
    epsilon = TRAINING_EP if for_training else TESTING_EP

    # Initialize game
    current_room_desc, current_quest_desc, terminal = framework.newGame()
    
    # Map text descriptions to state indices
    current_state_1 = dict_room_desc[current_room_desc]
    current_state_2 = dict_quest_desc[current_quest_desc]

    # For testing: accumulate discounted reward
    epi_reward = 0
    t = 0  # step counter for discounting

    while not terminal:
        # Select action using epsilon-greedy
        action_index, object_index = epsilon_greedy(current_state_1, current_state_2, q_func, epsilon)

        # Take a step in the environment
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index
        )

        # Map next state descriptions
        next_state_1 = dict_room_desc[next_room_desc]
        next_state_2 = dict_quest_desc[next_quest_desc]

        # TRAINING: update Q-function
        if for_training:
            tabular_q_learning(q_func, current_state_1, current_state_2, action_index, object_index,
                   reward, next_state_1, next_state_2, terminal)


        # TESTING: accumulate discounted reward
        if not for_training:
            epi_reward += (GAMMA ** t) * reward

        # Prepare for next step
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc
        current_state_1 = next_state_1
        current_state_2 = next_state_2
        t += 1

    if not for_training:
        return epi_reward

# pragma: coderesponse end


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
