# Ignore annoying warnings from imported envs
import warnings
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

import gym
import numpy as np

from .wrappers import *


def create_env(env_id: str, no_terminal: bool, env_time_limit: int, env_action_repeat: int):

    if env_id.startswith('MiniGrid-'):
        from .minigrid import MiniGrid
        env = MiniGrid(env_id)

    elif env_id.startswith('Atari-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)

    elif env_id.startswith('AtariGray-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat, grayscale=True)

    elif env_id.startswith('MiniWorld-'):
        import gym_miniworld.wrappers as wrap
        env = gym.make(env_id)
        env = wrap.DictWrapper(env)
        env = wrap.MapWrapper(env)
        # env = wrap.PixelMapWrapper(env)
        env = wrap.AgentPosWrapper(env)

    elif env_id.startswith('DmLab-'):
        from .dmlab import DmLab
        env = DmLab(env_id.split('-')[1].lower(), num_action_repeats=env_action_repeat)
        env = DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from .minerl import MineRL
        env = MineRL(env_id, np.load('data/minerl_action_centroids.npy'), action_repeat=env_action_repeat)

    elif env_id.startswith('DMC-'):
        from .dmc import DMC
        env = DMC(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)

    elif env_id == ("NavRep3DTrainEnv"):
        from navrep3d.navrep3dtrainenv import NavRep3DTrainEnvDiscrete
        class RGBImgPartialObsWrapper(gym.core.ObservationWrapper):
            """
            Wrapper for compatibility with dreamer
            """

            def __init__(self, env):
                super().__init__(env)

                _H = 64
                _W = 64

                self.observation_space = gym.spaces.Dict({
                    'image': gym.spaces.Box(low=0, high=255, shape=(_H, _W, 3), dtype='uint8'),
                    'mission': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
                })
                self.obs_space = self.observation_space

            def observation(self, obs):
                return {
                    'mission': obs[1],
                    'image': obs[0]
                }
        env = RGBImgPartialObsWrapper(NavRep3DTrainEnvDiscrete())

    else:
        env = gym.make(env_id)
        env = DictWrapper(env)

    if hasattr(env.action_space, 'n'):
        env = OneHotActionWrapper(env)
    if env_time_limit > 0:
        env = TimeLimitWrapper(env, env_time_limit)
    env = ActionRewardResetWrapper(env, no_terminal)
    env = CollectWrapper(env)
    return env
