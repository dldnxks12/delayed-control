import sys
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}

class InvertedSinglePendulumEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        self.History = 15

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        ob = np.reshape(ob, [1, -1])
        self.observation = np.hstack([ob] + [self.observation[0, :7*(self.History-1)].reshape([1,-1])])

        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 3.0) ** 2
        v1 = self.data.qvel[1]
        vel_penalty = 1e-3 * (v1 ** 2)

        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty

        if r <= 0 :
            r = 0
        else:
            r = (r / 10)

        terminated = False

        if self.render_mode == "human":
            self.render()

        return self.observation.copy()[0], r, terminated, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos[:1],  # cart x pos
                np.sin(self.data.qpos[1:]),  # link angles
                np.cos(self.data.qpos[1:]),
                np.clip(self.data.qvel, -10, 10),
                np.clip(self.data.qfrc_constraint, -10, 10),
            ]
        ).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + np.array([0, np.pi]) +
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        init_obs = np.reshape(self._get_obs(), [1, -1])
        self.observation = np.hstack([init_obs] * self.History)

        return self.observation.copy()[0]
