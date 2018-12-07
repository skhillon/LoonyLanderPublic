"""
Main file.
"""

# External Imports.
import gym
import time

# Project-level Imports.
from model import PolicyGradientNetwork

# Constants
_NUM_EPISODES = 5000
_FAIL_THRESHOLD = -250
_SEED = 1
_SOLVED_THRESHOLD = 200
_EPISODE_TIMEOUT = 120
_SEP = '=' * 25
_FORMATTER = '{} {}'

def setup_env():
    env = gym.make('LunarLander-v2').unwrapped
    env.seed(_SEED)
    print(_SEP)

    print(_FORMATTER.format('Action Space:', env.action_space))
    print(_FORMATTER.format('Observation Space:', env.observation_space))
    print(_FORMATTER.format('Upper Bound:', env.observation_space.high))
    print(_FORMATTER.format('Lower Bound:', env.observation_space.low))

    print(_SEP, end='\n\n')

    return env


def run(env, model):
    best_reward = float("-inf")
    start_time = time.perf_counter()

    for i in range(_NUM_EPISODES):
        observation = env.reset()
        total_reward = float("-inf")
        done = False

        while not done or not terminal_condition(total_reward, start_time):
            env.render()

            action = model.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            model.add_episode(observation, action, reward)
            observation = next_observation

            total_reward = model.get_total_reward()
            if total_reward > best_reward:
                best_reward = total_reward
                model.save_model()

        print_episode_info(i, start_time, total_reward, best_reward)
        model.learn()

    env.render()


def terminal_condition(total_reward, start):
    duration = time.perf_counter() - start
    return total_reward < _FAIL_THRESHOLD or duration > _EPISODE_TIMEOUT


def print_episode_info(episode, start_time, current_reward, highest_reward):
    elapsed_time = time.perf_counter() - start_time
    print(_SEP)

    print(_FORMATTER.format("Episode:", episode + 1))
    print(_FORMATTER.format("Time Elapsed:", elapsed_time))
    print(_FORMATTER.format("Episode Utility:", current_reward))
    print(_FORMATTER.format("Highest Utility So Far:", highest_reward))

    print(_SEP, end='\n\n')


if __name__ == '__main__':
    env = setup_env()
    model = PolicyGradientNetwork(env)
    run(env, model)
