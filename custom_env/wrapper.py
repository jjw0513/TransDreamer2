import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper



class MaxStepsWrapper(Wrapper):
    def __init__(self, env: gym.Env, max_steps: int, new_action_space = 3):
        super().__init__(env)
        self.max_steps = max_steps
        self.env.max_steps = max_steps
        self.env.env.max_steps = max_steps

        self.action_space = spaces.Discrete(new_action_space)
        self.env.action_space = self.action_space
        self.env.env.action_space = self.action_space

        self._env = env

    def sample_random_action(self):
        action = np.zeros((1, self._env.action_space.n,), dtype=float)
        idx = np.random.randint(0, self._env.action_space.n, size=(1,))[0]
        action[0, idx] = 1
        return action

    def reset(self, **kwargs):
        state = self.env.reset()
        if 'options' in kwargs and 'max_steps' in kwargs['options']:
            self.env.max_steps = kwargs['options']['max_steps']
        else:
            self.env.max_steps = self.max_steps
        state = self.preprocess_state(state[0])
        return state

    #def step(self, action):
    #    return self.env.step(action)
    def step(self, action):
        observation, r, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess_state(observation)
        return obs, r, terminated, truncated, info


    def preprocess_state(self, observation):
        if isinstance(observation, dict):
            return observation['image'].flatten()
        else:
            return observation.flatten()


class FullyCustom(FullyObsWrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self.env.max_steps = max_steps

    def reset(self, **kwargs):
        # Reset the environment and update the max_steps
        state = super().reset()
        if 'options' in kwargs and 'max_steps' in kwargs['options']:
            self.env.max_steps = kwargs['options']['max_steps']
        else:
            self.env.max_steps = self.max_steps
        return state

class ActionSpaceWrapper(Wrapper):
    def __init__(self, env: gym.Env,  max_steps, new_action_space: int):
        super().__init__(env)
        self.new_action_space = new_action_space
        self.action_space = spaces.Discrete(new_action_space)
        self.env.action_space = self.action_space


        self.max_steps = max_steps
        self.env.max_steps = max_steps

    def reset(self, **kwargs):
        # Reset the environment and update the max_steps
        state = super().reset()



        self.env.max_steps = self.max_steps
        return state