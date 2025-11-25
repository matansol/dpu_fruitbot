import unittest
import os
import numpy as np
from stable_baselines3 import PPO
import gym, procgen
from procgen.wrappers import make_fruitbot_basic
import imageio
from PIL import Image

# Print wrapper chain for debug
def print_wrapper_chain(env):
    i = 0
    while hasattr(env, 'env'):
        print(f"Wrapper {i}: {type(env).__name__}")
        env = env.env
        i += 1
    print(f"Base env: {type(env).__name__}")


class TestSaveObs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.screenshots_dir = "screenshots"
        os.makedirs(cls.screenshots_dir, exist_ok=True)
        cls.model_path = "models/fruitbot/20251123-133459_easy/ppo_final.zip"

    def setUp(self):
        self.env = gym.make(
            'procgen-fruitbot-v0', 
            render_mode='rgb_array',
            distribution_mode='easy',
            use_discrete_action_wrapper=True,
            use_stay_bonus_wrapper=True,
            stay_bonus=0.1,
        )
        
        print_wrapper_chain(self.env)

        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
            print(f"Loaded PPO model from {self.model_path}")
        else:
            self.model = None
            print("Using random actions")

    def tearDown(self):
        self.env.close()

    def test_action_space(self):
        print("action_space test: =========================================================")
        print(self.env.action_space)
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        

    def test_save_obs_and_video(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        print(f"Observation shape: {obs.shape}")
        print(f"Observation dtype: {obs.dtype}")

        frames = []
        frame = self.env.render()
        if frame is not None:
            print(f"Rendered frame shape: {frame.shape}")
            frames.append(frame)
        self.assertIsNotNone(frame, "Initial frame should not be None")

        # Run a few steps and capture high-res frames
        for step in range(5):
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            result = self.env.step(action)
            obs, rew, done, info = result
            frame = self.env.render()
            if frame is not None:
                frames.append(frame)
            if done:
                obs = self.env.reset()
                

        # Validate frames before saving
        valid_frames = []
        for f in frames:
            arr = np.asarray(f)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                valid_frames.append(arr)
            else:
                print(f"Skipping invalid frame with shape {arr.shape}")

        # Save first frame as image
        if valid_frames:
            img_path = os.path.join(self.screenshots_dir, "fruitbot_highres_frame.png")
            Image.fromarray(valid_frames[0]).save(img_path)
            print(f"Saved high-res screenshot to {img_path}")
            self.assertTrue(os.path.exists(img_path), "Screenshot file should exist")
        else:
            print("No frames captured!")
            self.fail("No frames captured!")



if __name__ == '__main__':
    unittest.main()