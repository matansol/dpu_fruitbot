import unittest
import os
import numpy as np
import cv2  # Add this import at the top
from stable_baselines3 import PPO
import gym, procgen
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
        # cls.screenshots_dir = "screenshots"
        cls.screenshots_dir = "tests/frameshots"
        os.makedirs(cls.screenshots_dir, exist_ok=True)
        cls.model_path = "models\\fruitbot\\20251127-002723_easy\\ppo_final.zip"

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
        done = False
        while not done:
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            result = self.env.step(action)
            obs, rew, done, info = result
            frame = self.env.render()
            if not done and frame is not None:
                frames.append(frame)

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

        print(f"Captured {len(valid_frames)} valid frames out of {len(frames)} total frames")
        
        # Save frames as images
        if valid_frames:
            valid_frames = valid_frames + [valid_frames[-1]]*3  # Ensure at least one frame
            for idx, frame in enumerate(valid_frames):
                if idx % 3 == 0:  # Save every 3rd frame
                    img_path = os.path.join(self.screenshots_dir, f"fruitbot_frame_{idx}.png")
                    Image.fromarray(frame).save(img_path)
                    print(f"Saved high-res screenshot to {img_path}")
                    self.assertTrue(os.path.exists(img_path), f"Screenshot file {img_path} should exist")
        else:
            print("No frames captured!")
            self.fail("No frames captured!")

        # Save video using OpenCV
        if valid_frames:
            video_path = os.path.join(self.screenshots_dir, "fruitbot_run.mp4")
            height, width, _ = valid_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 15.0, (width, height))
            
            for frame in valid_frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
            print(f"Saved video to {video_path}")
            self.assertTrue(os.path.exists(video_path), f"Video file {video_path} should exist")
        else:
            print("No frames captured!")
            self.fail("No frames captured!")



if __name__ == '__main__':
    unittest.main()