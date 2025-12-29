import os

import dpu_clf
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import numpy
import numpy as np
import json
from datetime import datetime
import random
import sys
from PIL import Image

# Add procgen imports
import gym
import procgen
from procgen import gym_registration  # Explicitly register procgen envs

# Add multiagent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multiagent'))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import socketio
import asyncio

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

# Import refactored dpu_clf functions
from dpu_clf import *

from functools import reduce
from typing import Dict, Any, Optional, Tuple, List, Union

# Procgen Fruitbot action constants
# Procgen uses 15 discrete actions (standard for most Procgen games)
FRUITBOT_ACTIONS = {
    'STAY': 1,
    'LEFT': 0,
    'RIGHT': 2,
    'THROW': 3,
    }

# ---------------------- ENV & DATABASE SETUP ----------------------

load_dotenv()

# FastAPI application
app = FastAPI()

# Socket.IO configuration
sio_config = {
    "async_mode": "asgi",
    "cors_allowed_origins": "*",
    "logger": False,
    "engineio_logger": False,
    "ping_timeout": 60,
    "ping_interval": 25,
    "transports": ['websocket'],
    "allow_upgrades": False,
    "http_compression": False,
    "compression": False,
    "max_http_buffer_size": 2000000,
    "max_connections": 100,
    "always_connect": True
}

sio = socketio.AsyncServer(**sio_config)

# Middleware for WebSocket-only
@app.middleware("http")
async def reject_polling_middleware(request: Request, call_next):
    """Reject any Socket.IO polling requests to force WebSocket-only connections"""
    if (request.url.path.startswith("/socket.io/") and 
        request.query_params.get("transport") == "polling"):
        print(f"REJECTED POLLING REQUEST: {request.url}")
        from fastapi.responses import Response
        return Response("WebSocket-only mode: Polling transport disabled", status_code=400)
    
    response = await call_next(request)
    return response

# Wrap the FastAPI app with Socket.IO's ASGI application
app.mount("/static", StaticFiles(directory="static"), name="static")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Global variable to control database saving
save_to_db = False

# SQLAlchemy setup - only connect if save_to_db is enabled
Base = declarative_base()
engine = None
SessionLocal = None

if save_to_db:
    DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")
    print(f"Initializing database connection...")
    try:
        engine = create_engine(DATABASE_URI, echo=False)
        SessionLocal = sessionmaker(bind=engine)
        print(f"Database connection initialized successfully")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print(f"Continuing without database support...")
        save_to_db = False
else:
    print("Database saving is disabled (save_to_db=False)")

# class Action(Base):
#     __tablename__ = "actions"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String(100))
#     action_type = Column(String(20))
#     agent_action = Column(Boolean)
#     score = Column(Float)
#     reward = Column(Float)
#     done = Column(Boolean)
#     episode = Column(Integer)
#     timestamp = Column(String(30))
#     agent_index = Column(Integer)
#     env_state = Column(String(1000))

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    timestamp = Column(String(30))
    similarity_level = Column(Integer)
    final_score = Column(Float, default=0.0)  # Default to 0.0 if not set


class FeedbackAction(Base):
    __tablename__ = "feedback_actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    env_state = Column(String(1000), nullable=True)
    agent_action = Column(String(20))
    feedback_action = Column(String(20))
    feedback_explanation = Column(String(500), nullable=True)
    action_index = Column(Integer)
    timestamp = Column(String(30))  
    episode_index = Column(Integer)
    agent_name = Column(String(100))
    similarity_level = Column(Integer)
    env_seed = Column(Integer, nullable=True)

class UserChoice(Base): 
    __tablename__ = "user_choices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    old_agent_name = Column(String(100)) 
    new_agent_name = Column(String(100))
    timestamp = Column(String(50))
    demonstration_time = Column(String(50))
    episode_index = Column(Integer)
    choice_to_update = Column(Boolean)
    choice_explanation = Column(String(500), nullable=True)
    similarity_level = Column(Integer)
    feedback_score = Column(Float, nullable=True)
    feedback_count = Column(Integer, nullable=True)
    env_seed_feedback = Column(Integer, nullable=True)
    env_seed_demonstration = Column(Integer, nullable=True)

def clear_database() -> None:
    """Clears the database tables."""
    if not save_to_db or engine is None:
        print("Database is disabled, skipping clear_database")
        return
    print("Clearing database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def create_database() -> None:
    """Creates the database tables if they do not already exist."""
    if not save_to_db or engine is None:
        print("Database is disabled, skipping create_database")
        return
    print("Ensuring database tables are created...")
    Base.metadata.create_all(bind=engine)


# Helper
async def in_thread(func: callable, *args, **kw) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kw))


class GameControl:
    def __init__(
        self,
        models_paths: Dict[int, Dict[str, Any]],
        models_distance: Dict[int, Dict[int, Dict[str, Any]]], # {model_i: {target_model_i: {'name': str, 'configs': List[(seed, config_index)]}}}
        user_id: str,
        similar_level_env: int = 0,
        feedback_partial_view: bool = True,
        env_seed_list: List[int] = list(range(100, 1000)),
    ) -> None:
        self.agent_index = 0  # Start with first agent
        self.models_paths = models_paths
        self.models_distance = models_distance
        self.episode_num = 0
        self.scores_lst = []
        self.last_obs = None
        self.episode_actions = []
        self.agent_last_pos = None
        self.episode_images = []
        self.episode_frames = []  # Store raw RGB frames for video creation
        self.episode_obs = []
        self.episode_agent_locations = []
        self.user_id = user_id
        self.wall_penalty: float = -3.0
        self.last_score: float = 0.0
        self.similar_level_env: int = int(similar_level_env)
        self.ppo_agent = None
        self.prev_agent = None
        self.current_agent_path = ""  
        self.prev_agent_path = ""
        self.prev_agent_index: int = -1
        self.current_obs = {}
        self.feedback_partial_view: bool = feedback_partial_view
        self.feedback_score: int = 0 # the number of good feedbacks the user gave
        self.number_of_feedbacks: int = 0 # total number of feedbacks the user gave
        self.board_seen: list = []
        self.demonstration_unique_envs: list = []  # List to store unique environments for demonstrations
        self.prev_agent_score_list: list = []
        self.current_agent_score_list: list = []
        self.past_choices: set = set()  # To avoid repeating the same choice
        self.step_count: int = 0  # Track step count within episode
        self.env_seed_demonstration: int = 0  # Seed for demonstration environment
        self.env_seed_list: list = env_seed_list  # List of seeds for environments
        self.env_seed_used: list = []  # List of used environments
        self.current_config_index: int = 0
        
        # Define 4 different environment configurations
        self.env_configs = [
            # Config 1: No walls, balanced fruits and bad items
            {
                'name': 'No Walls - Balanced',
                'fruitbot_num_walls': 1,
                'fruitbot_num_good_min': 5,
                'fruitbot_num_good_range': 5,
                'fruitbot_num_bad_min': 5,
                'fruitbot_num_bad_range': 5,
                'fruitbot_wall_gap_pct': 90,
                'fruitbot_door_prob_pct': 0,
                'food_diversity': 4,
                'use_discrete_action_wrapper': True,
                'use_stay_bonus_wrapper': False,
            },
            # Config 2: With walls, only good fruits
            {
                'name': 'Walls - Only Fruits',
                'fruitbot_num_walls': 3,
                'fruitbot_num_good_min': 6,
                'fruitbot_num_good_range': 4,
                'fruitbot_num_bad_min': 1,
                'fruitbot_num_bad_range': 0,
                'fruitbot_wall_gap_pct': 30,
                'fruitbot_door_prob_pct': 0,
                'food_diversity': 4,
                'use_discrete_action_wrapper': True,
                'use_stay_bonus_wrapper': False,
            },
            # Config 3: With walls, few fruits, lots of bad items
            {
                'name': 'Walls - Sparse Fruits Many Bad',
                'fruitbot_num_walls': 4,
                'fruitbot_num_good_min': 2,
                'fruitbot_num_good_range': 3,
                'fruitbot_num_bad_min': 10,
                'fruitbot_num_bad_range': 5,
                'fruitbot_wall_gap_pct': 40,
                'fruitbot_door_prob_pct': 0,
                'food_diversity': 4,
                'use_discrete_action_wrapper': True,
                'use_stay_bonus_wrapper': False,
            },
            # Config 4: With walls, doors, and all food types
            {
                'name': 'Walls + Doors - Full Complexity',
                'fruitbot_num_walls': 5,
                'fruitbot_num_good_min': 5,
                'fruitbot_num_good_range': 5,
                'fruitbot_num_bad_min': 5,
                'fruitbot_num_bad_range': 5,
                'fruitbot_wall_gap_pct': 50,
                'fruitbot_door_prob_pct': 20,
                'food_diversity': 6,
                'use_discrete_action_wrapper': True,
                'use_stay_bonus_wrapper': False,
            }
        ]

    def create_new_env(self, env_seed: int = 0, config_index = None) -> Tuple[gym.Env, int, str]:
        """Create a new environment instance with specified configuration.
        
        Args:
            env_seed: Random seed for environment generation
            config_index: Index of configuration to use (0-3), or None for random selection
        
        Returns:
            Configured gym environment
        """
        # Select random config if not specified
        if config_index is None:
            config_index = random.randint(0, len(self.env_configs) - 1)
        
        # Get config and make a copy to avoid modifying the original
        config = self.env_configs[config_index].copy()
        config_name = config.pop('name')  # Remove name from kwargs
        
        print(f"[create_new_env] Using config: {config_name} (index {config_index})")
        
        seed = env_seed if env_seed else self.env_seed
        env = gym.make("procgen-fruitbot-v0", distribution_mode="easy", rand_seed=seed, **config)
        return env, config_index, config_name
        
    @timeit
    def reset(self) -> np.ndarray:
        optinal_seed = [seed for seed in self.env_seed_list if seed not in self.env_seed_used]
        # Set the environment FIRST before calling update_agent
        self.env_seed = random.choice(optinal_seed)
        self.env_seed_used.append(self.env_seed)
        
        # Create new environment with randomly selected configuration
        self.env, self.current_config_index, self.current_config_name = self.create_new_env(self.env_seed)
        print(f"[reset] Created new environment with seed {self.env_seed}, config: {self.current_config_name}")
        
        # Now safe to call update_agent since self.env exists
        self.update_agent(None, None)
        
        # Old Gym API returns only observation, not (obs, info)
        obs = self.env.reset()
        
        # Store observation (Procgen returns RGB array directly)
        print(f"[reset] resetting the episode_obs list")
        self.episode_obs = [obs]
        
        # Initialize collision tracking for this episode
        self.collision_positions = []  # List of {x, y, type, pixel_x, pixel_y}
        
        # Procgen doesn't expose position, track as None
        self.episode_agent_locations = [(None, None)]
        
        self.feedback_score = 0
        # self.saved_env = copy.deepcopy(self.env)
        
        self.step_count = 0  # Reset step count
        self.score = 0
        
        # Reset frame storage
        self.episode_frames = []
        
        return obs

    # @timeit
    def step(self, action: int, agent_action: bool = False) -> Dict[str, Any]:
        # Old Gym API returns 4 values, new Gym/Gymnasium returns 5
        t_start = time.time()
        result = self.env.step(action)
        t_env_step = time.time() - t_start
        
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            observation, reward, done, info = result
        else:
            raise ValueError(f"Unexpected step() return length: {len(result)}")

        # Convert numpy types to Python native types for JSON serialization
        reward = float(reward)
        self.score += reward
        self.score = float(round(self.score, 1))
        if done:
            self.episode_actions.append(int(action))
            self.scores_lst.append(self.score)
            return {
                'image': None,
                'episode': self.episode_num,
                'reward': reward,
                'done': done,
                'score': self.score,
                'agent_action': agent_action,
                'step_count': self.step_count,
                'agent_index': self.agent_index
            }
        
        # All actions are valid, just add to episode
        self.episode_actions.append(int(action))

        self.episode_obs.append(observation)
        
        # Get RGB image from info dict (high-res rendering from InfoRgbRenderWrapper)
        t0 = time.time()
        img_frame = info.get('rgb', observation)
        
        # Save raw frame for video creation
        self.episode_frames.append(img_frame.copy())
        t_frame_copy = time.time() - t0

                # Track collision positions with pixel conversion
        if info['collision_type'] > 0 and not done:
            collision_x = float(info['collision_x'])
            collision_y = float(info['collision_y'])
            collision_type = int(info['collision_type'])
            agent_y = 0.9
            
            # Store collision data with world coordinates
            collision_data = {
                'world_x': collision_x,
                'world_y': collision_y,
                'agent_y': agent_y,
                'type': collision_type,
                'step': self.step_count
            }
            self.collision_positions.append(collision_data)
            
            # print(f"Collision at world({collision_x:.2f}, {collision_y:.2f}), type={collision_type}")
        
        t0 = time.time()
        image_base64 = image_to_base64(img_frame)
        t_image_encode = time.time() - t0
            
            
            # Entity types:
            # 1 = BARRIER (wall)
            # 4 = BAD_OBJ (bad food)
            # 7 = GOOD_OBJ (good food)
            # 10 = LOCKED_DOOR
            # 12 = PRESENT (goal)
            
        self.episode_images.append(image_base64)

        self.current_obs = observation
        self.step_count += 1
        
        # Log timing every 50 steps
        if self.step_count % 50 == 0:
            t_total = time.time() - t_start
            print(f"[step] Step {self.step_count} timing: env_step={t_env_step*1000:.1f}ms, frame_copy={t_frame_copy*1000:.1f}ms, image_encode={t_image_encode*1000:.1f}ms, total={t_total*1000:.1f}ms")
                
        return {
            'image': image_base64,
            'episode': self.episode_num,
            'reward': reward,
            'done': done,
            'score': self.score,
            'agent_action': agent_action,
            'step_count': self.step_count,
            'agent_index': self.agent_index
        }

    def handle_action(self, action_str: str) -> Dict[str, Any]:
        """Map keyboard input to Fruitbot actions."""
        key_to_action = {
            "ArrowLeft": FRUITBOT_ACTIONS['LEFT'],
            "ArrowRight": FRUITBOT_ACTIONS['RIGHT'],
            "ArrowUp": FRUITBOT_ACTIONS['STAY'],
            "ArrowDown": FRUITBOT_ACTIONS['STAY'],
            "Space": FRUITBOT_ACTIONS['THROW'],
        }
        action = key_to_action.get(action_str, FRUITBOT_ACTIONS['STAY'])
        return self.step(action)

    @timeit
    def get_initial_observation(self) -> Dict[str, Any]:
        try:
            print(f"[get_initial_observation] Starting reset...")
            self.current_obs = self.reset()
            self.agent_last_pos = None
            self.episode_actions = []
            
            print(f"[get_initial_observation] Reset complete, obs shape: {self.current_obs.shape if hasattr(self.current_obs, 'shape') else 'N/A'}")
            
            try:
                print(f"[get_initial_observation] Getting initial frame from step with NOOP action...")
                # Take a NOOP step to get the first info dict with 'rgb'
                result = self.env.step(1)  # NOOP/STAY action
                if len(result) == 4:
                    obs, reward, done, info = result
                else:
                    obs, reward, terminated, truncated, info = result
                
                # Get frame from info['rgb']
                frame = info.get('rgb', None)
                
                if frame is None:
                    print("[get_initial_observation] WARNING: 'rgb' not in info, using observation")
                    frame = obs
                
                print(f"[get_initial_observation] Got initial frame, shape: {frame.shape}")
                
            except Exception as render_error:
                print(f"[get_initial_observation] ERROR getting initial frame: {render_error}")
                import traceback
                traceback.print_exc()
                raise
            
            # Save initial frame
            self.episode_frames.append(frame.copy())
            
            print(f"[get_initial_observation] Converting frame to base64...")
            # Convert directly to base64 with resizing
            image_base64 = image_to_base64(frame, resize=(512, 512))
            self.episode_images = [image_base64]
            self.episode_num += 1
            
            step_count = 0
            
            print(f"User {self.user_id} Episode {self.episode_num} started {'_'*100}")
            return {
                'image': image_base64,
                'last_score': self.last_score,
                'action': None,
                'reward': 0,
                'done': False,
                'score': 0,
                'episode': self.episode_num,
                'agent_action': False,
                'step_count': step_count,
                'agent_index': self.agent_index,
            }
        except Exception as e:
            print(f"[get_initial_observation] FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def agent_action(self) -> Dict[str, Any]:
        # Get action from PPO agent
        # agent_config = self.models_paths[self.agent_index]
        
        # PPO agent: predict returns (action, _states)
        action, _ = self.ppo_agent.predict(self.current_obs, deterministic=True)
        action = action.item() if hasattr(action, 'item') else int(action)
        
        result = self.step(action, True)
        result['action'] = action
        return result

    def revert_to_old_agent(self) -> None:
        self.ppo_agent = self.prev_agent
        self.agent_index = self.prev_agent_index
        self.current_agent_path = self.models_paths[self.agent_index]['path']
        print(f'(revert) prev agent index={self.prev_agent_index}, prev agent path={self.prev_agent_path}')
        print(f"(revert) current agent index={self.agent_index}")

    def count_similar_actions(
        self,
        other_agent: Any,
        other_agent_config: Dict[str, Any],
        feedback_indexes: List[int]
    ) -> int:
        similar_actions = 0
        
        for i, action in enumerate(self.episode_actions):
            if i in feedback_indexes:
                continue
            saved_obs = self.episode_obs[i]
            
            # Get predicted action from other agent
            predicted_action = other_agent.predict(saved_obs, deterministic=True)
            predicted_action = predicted_action[0]
            predicted_action = predicted_action.item() if hasattr(predicted_action, 'item') else int(predicted_action)
            
            if action == predicted_action:
                similar_actions += 1
        return similar_actions

    @timeit
    def update_agent(self, data: Optional[Dict[str, Any]], sid: Optional[str]) -> Optional[bool]:
        print(f"\n{'='*80}")
        print(f"[update_agent] CALLED - Starting agent update process")
        print(f"[update_agent] User ID: {self.user_id}")
        print(f"[update_agent] Current agent index: {self.agent_index}")
        print(f"[update_agent] Current agent path: {self.current_agent_path}")
        print(f"[update_agent] Data received: {data is not None}")
        print(f"[update_agent] observations stored: {len(self.episode_obs)}")
        
        if self.ppo_agent is None:
            agent_config = self.models_paths[self.agent_index]
            self.ppo_agent = load_agent(self.env, agent_config['path'])
            self.current_agent_path = agent_config['path']
            self.prev_agent = self.ppo_agent
            self.prev_agent_path = self.current_agent_path
            self.prev_agent_index = self.agent_index
            print(f'[update_agent] Loaded first model: {agent_config["name"]}')
            print(f'[update_agent] Initial agent index: {self.agent_index}')
            return None
            
        if data is None:
            print("[update_agent] Data is None, returning")
            return None
            
        if data.get('updateAgent', False) == False:
            print("[update_agent] updateAgent flag is False, returning")
            return None
            
        user_feedback = data.get('userFeedback')
        if user_feedback is None or len(user_feedback) == 0:
            print("[update_agent] No user feedback provided, returning")
            return None
                
        # Remove duplicates
        # unique_feedback = {}
        # for feedback in user_feedback:
        #     unique_feedback[feedback['index']] = feedback
        # user_feedback = list(unique_feedback.values())
        self.number_of_feedbacks = len(user_feedback)
        
        print(f"[update_agent] After deduplication: {len(user_feedback)} unique feedback items")

        # Save feedback to DB
        if save_to_db and sid:
            print(f"[update_agent] Saving feedback to database...")
            for action_feedback in user_feedback:
                try:
                    session = SessionLocal()
                    action_index = action_feedback['index']
                    if action_index >= len(self.episode_obs):
                        obs_str = "No observation available"
                    else:
                        obs = self.episode_obs[action_index]
                        # For Procgen, obs is already an RGB array
                        obs_str = json.dumps(obs.tolist() if isinstance(obs, numpy.ndarray) else str(obs))[:1000]
                    
                    agent_action = action_feedback['agent_action']
                    feedback_action = FeedbackAction(
                        user_id=self.user_id,
                        env_state=None, #obs_str[:1000],  # Truncate to fit column
                        agent_action=str(agent_action),
                        feedback_action=str(action_feedback['feedback_action']),
                        feedback_explanation=action_feedback.get('feedback_explanation', ''),
                        action_index=action_index,
                        episode_index=self.episode_num,
                        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        agent_name=self.models_paths[self.agent_index]['name'],
                        similarity_level=self.similar_level_env,
                        env_seed=self.env_seed,
                    )
                    session.add(feedback_action)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Database operation failed: {e}")
                finally:
                    session.close()

        feedback_indexes = [feedback['index'] for feedback in user_feedback]
        print(f"[update_agent] Feedback indexes: {feedback_indexes}")

        optimal_agents = []
        target_models_indexes = self.models_distance[self.agent_index]
        print(f"\n[update_agent] Evaluating {len(target_models_indexes)} candidate agents...")
        
        for model_i, agent_info in target_models_indexes.items():
            model_name = agent_info['name']
            agent_data = self.models_paths[model_i]
            path = agent_data['path']
            agent = load_agent(self.env, path)

            agent_correctness = 0
            for action_feedback in user_feedback:
                if action_feedback['index'] >= len(self.episode_obs):
                    print(f"[update_agent] WARNING: Feedback index {action_feedback['index']} out of bounds")
                    continue
                saved_obs = self.episode_obs[action_feedback['index']]
                
                agent_predict_action = agent.predict(saved_obs, deterministic=True)
                agent_predict_action = int(agent_predict_action[0].item() if hasattr(agent_predict_action[0], 'item') else agent_predict_action[0])
                
                if agent_predict_action == int(action_feedback['feedback_action']):
                    agent_correctness += 1

            if agent_correctness > 0:
                similar_actions = self.count_similar_actions(agent, agent_data, feedback_indexes)
                print(f"[update_agent] Agent {model_name} similar actions: {similar_actions}")
                optimal_agents.append({
                    "agent": agent,
                    "name": model_name,
                    "path": path,
                    "correctness_feedback": agent_correctness,
                    "similar_actions": similar_actions,
                    "model_index": model_i
                })

        print(f"\n{'_'*80}")
        print(f"[update_agent] CANDIDATE AGENTS SUMMARY:")
        for agent_dict in optimal_agents:
            print(f"  - {agent_dict['name']}: correctness={agent_dict['correctness_feedback']}, similar={agent_dict['similar_actions']}")
        print(f"{'_'*80}\n")

        if len(optimal_agents) == 0:
            print("[update_agent] WARNING: No optimal agents found, using fallback")
            optimal_agents.append({
                "agent": agent,
                "path": path,
                "name": model_name,
                "correctness_feedback": agent_correctness,
                "model_index": model_i
            })

        def agent_cmp(a, b):
            if a["correctness_feedback"] > b["correctness_feedback"]:
                return a
            elif a["correctness_feedback"] < b["correctness_feedback"]:
                return b
            else:
                return a if a["similar_actions"] >= b["similar_actions"] else b

        new_agent_dict = reduce(agent_cmp, optimal_agents)
        
        print(f"\n[update_agent] AGENT SELECTION:")
        print(f"  Current agent: {self.agent_index} - {self.models_paths[self.agent_index]['name']}")
        print(f"  Selected agent: {new_agent_dict['model_index']} - {new_agent_dict['name']}")
        print(f"  Correctness: {new_agent_dict['correctness_feedback']}")
        
        self.prev_agent = self.ppo_agent
        self.prev_agent_path = self.current_agent_path
        self.prev_agent_index = self.agent_index
        self.ppo_agent = new_agent_dict["agent"]
        self.agent_index = new_agent_dict["model_index"]
        self.current_agent_path = self.models_paths[self.agent_index]['path']
        
        print(f"\n[update_agent] UPDATE COMPLETE:")
        print(f"  Previous agent index: {self.prev_agent_index}")
        print(f"  Current agent index: {self.agent_index}")
        print(f"  Agent changed: {self.prev_agent_index != self.agent_index}")
        print(f"{'='*80}\n")
        
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent
        return True

    @timeit
    def agents_different_routs(
        self,
        similarity_level: int = 5,
    ) -> Dict[str, Any]:
        print(f"\n{'='*80}")
        print(f"[agents_different_routs] CALLED - Generating agent comparison paths")
        print(f"[agents_different_routs] User ID: {self.user_id}")
        print(f"[agents_different_routs] Similarity level: {similarity_level}")
        print(f"[agents_different_routs] Current agent index: {self.agent_index}")
        print(f"[agents_different_routs] Previous agent index: {self.prev_agent_index}")
        print(f"{'='*80}\n")
        
        if self.ppo_agent == None or self.prev_agent == None:
            print(f"[agents_different_routs] ERROR: Missing agents")
            print(f"  ppo_agent exists: {self.ppo_agent is not None}")
            print(f"  prev_agent exists: {self.prev_agent is not None}")
            return {}

        env_kwargs = {
            'fruitbot_num_walls': 5,
            'fruitbot_num_good_min': 10,
            'fruitbot_num_good_range': 1,
            'fruitbot_num_bad_min': 10,
            'fruitbot_num_bad_range': 1,
            'fruitbot_wall_gap_pct': 50,
            'fruitbot_door_prob_pct': 0,
            'food_diversity': 4,
            "use_discrete_action_wrapper": True, 
            "use_stay_bonus_wrapper": False
        }
        if similarity_level == 0:
            print(f"[agents_different_routs] Similarity level 0 selected, returning empty result")
            return {}

        if similarity_level == 1: # same env
            self.env_seed_demonstration = self.env_seed  # Same seed for demonstration
            config_idx = self.current_config_index
        elif similarity_level == 2: # random env
            self.env_seed_demonstration = random.choice(self.env_seed_list)
            config_idx = random.randint(0, len(self.env_configs) - 1)
        else: # contrst
            # Access the configs for the target agent directly from nested dict
            config_options = self.models_distance[self.prev_agent_index][self.agent_index]['configs']
            print(f"[agents_different_routs] Config options for similarity level 2: {config_options}")
            self.env_seed_demonstration, config_idx = random.choice(config_options)
            print(f"[agents_different_routs] Selected seed {self.env_seed_demonstration} and config index {config_idx} for similarity level 2")
        
        self.env_seed_used.append(self.env_seed_demonstration)

        print(f"[agents_different_routs] Initializing env1 and env2 with seed {self.env_seed_demonstration}... similarity_level={similarity_level}")
        
        # Choose a random config for this comparison
        
        env1, _, config_name = self.create_new_env(self.env_seed_demonstration, config_index=config_idx)
        env2, _, _ = self.create_new_env(self.env_seed_demonstration, config_index=config_idx)
        print(f"[agents_different_routs] Using config: {config_name} for both agents")
        # env1 = gym.make("procgen-fruitbot-v0", render_mode="rgb_array", num_levels=0, start_level=0, distribution_mode="easy", rand_seed=self.env_seed_demonstration, **env_kwargs)  
        # env2 = gym.make("procgen-fruitbot-v0", render_mode="rgb_array", num_levels=0, start_level=0, distribution_mode="easy", rand_seed=self.env_seed_demonstration, **env_kwargs)  
        # env1 = gym.make("procgen:procgen-fruitbot-v0", render_mode="rgb_array", num_levels=0, start_level=0, distribution_mode="easy", **kwargs)  
        # env2 = gym.make("procgen:procgen-fruitbot-v0", render_mode="rgb_array", num_levels=0, start_level=0, distribution_mode="easy", **kwargs)  

        frames_list_updated, frames_indexes1, collect_indexes1, wall_collision_index1, collisions1 = dpu_clf.record_frames(env1, self.ppo_agent, frames_jumps=5)
        frames_list_prev, frames_indexes2, collect_indexes2, wall_collision_index2, collisions2 = dpu_clf.record_frames(env2, self.prev_agent, frames_jumps=5)
        
        img_updated, _ = dpu_clf.draw_full_path(frames_list_updated, frames_indexes=frames_indexes1, collect_indexes=collect_indexes1, collisions=collisions1, frames_jumps=5, wall_collision_index=wall_collision_index1, use_rectangle=True)
        print(f"[agents_different_routs] Path image 1 size: {img_updated.size if img_updated else 'None'}")
        
        img_prev, _ = dpu_clf.draw_full_path(frames_list_prev, frames_indexes=frames_indexes2, collect_indexes=collect_indexes2, collisions=collisions2, frames_jumps=5, wall_collision_index=wall_collision_index2, use_rectangle=True)
        print(f"[agents_different_routs] Path image 2 size: {img_prev.size if img_prev else 'None'}")

        self.past_choices.add((self.models_paths[self.prev_agent_index]['name'], self.models_paths[self.agent_index]['name']))
        
        # Send images at original size without resizing
        image_updated_base64 = image_to_base64(img_updated, resize=None)
        image_prev_base64 = image_to_base64(img_prev, resize=None)
        print(f"[agents_different_routs] Base64 image 1 length: {len(image_updated_base64)}")
        print(f"[agents_different_routs] Base64 image 2 length: {len(image_prev_base64)}")
        
        print(f"\n[agents_different_routs] COMPLETE - Returning comparison data")
        print(f"{'='*80}\n")
        
        return {
            'rawImageUpdated': image_updated_base64,
            'rawImagePrev': image_prev_base64,
            # 'prevMoveSequence': [],
            # 'updatedMoveSequence': [],
        }


    def save_user_choice(
            self,
            user_choice: bool,
            explanation: str,
            demonstration_time_fmt: str,
        ) -> None:
        """Save the user's choice between agents to the database."""
        if not save_to_db:
            print("[save_user_choice] Database saving is disabled.")
            return
        try:
            session = SessionLocal()
            user_choice_entry = UserChoice(
                user_id=self.user_id,
                old_agent_name=self.models_paths[self.prev_agent_index]['name'],
                new_agent_name=self.models_paths[self.agent_index]['name'],
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                demonstration_time=demonstration_time_fmt,
                episode_index=self.episode_num,
                choice_to_update=user_choice,
                choice_explanation=explanation,
                similarity_level=self.similar_level_env,
                feedback_score=self.feedback_score,
                feedback_count=self.number_of_feedbacks,
                env_seed_feedback=self.env_seed,
                env_seed_demonstration=self.env_seed_demonstration if self.similar_level_env >= 1 else None,
            )
            session.add(user_choice_entry)
            session.commit()
        
        except Exception as e:
            print(f"[save_user_choice] ERROR: Could not create database session: {e}")
            session.rollback()
            return
        finally:
            session.close()
        
# ---------------- Global Variables ----------------

game_controls: Dict[str, GameControl] = {}
sid_to_user: Dict[str, str] = {}

# Procgen Fruitbot models configuration

# models\fruitbot\20251213-212435_easy\ppo_final.zip - לא נתקע ולא מתאמץ לקחת או מתחמק
# נתקע לא מעט, מנסה להתחמק מהכל - models\fruitbot\20251203-132922_easy\ppo_final.zip

models_dict = {
    0: {'path': "models/fruitbot/20251222-161336_easy/ppo_final.zip", 'name': 'Agent0'},
    1: {'path': "models/fruitbot/20251225-083104_easy/ppo_final.zip", 'name': 'Agent1'},
    2: {'path': "models/fruitbot/20251224-140335_easy/ppo_final.zip", 'name': 'Agent2'},
    3: {'path': "models/fruitbot/20251130-001800_easy/ppo_final.zip", 'name': 'Agent3'},
    4: {'path': "models/fruitbot/20251203-132922_easy/ppo_final.zip", 'name': 'Agent4'}, # avoid all
    5: {'path': "models/fruitbot/20251224-133036_easy/ppo_final.zip", 'name': 'Agent5'}, 
}

# for each model index a list of optinal ather agents to switch to, with the other model index, name, and list of (env_seeds, env_config_index)
models_distance = {
    0: {
        1: {'name': 'Agent1', 'configs': [(12343, 0), (15678, 2)]},
        2: {'name': 'Agent2', 'configs': [(10234, 1), (18456, 3), (14567, 0)]},
        3: {'name': 'Agent3', 'configs': [(11890, 2), (16234, 1)]},
        4: {'name': 'Agent4', 'configs': [(13456, 3), (17890, 0), (19123, 2)]}
    },
    1: {
        0: {'name': 'Agent0', 'configs': [(10567, 1), (15234, 3)]},
        2: {'name': 'Agent2', 'configs': [(12789, 0), (16890, 2), (18234, 1)]},
        3: {'name': 'Agent3', 'configs': [(14123, 2), (17456, 0)]},
        5: {'name': 'Agent5', 'configs': [(11234, 3), (19567, 1)]}
    },
    2: {
        0: {'name': 'Agent0', 'configs': [(10890, 2), (15890, 0), (18567, 3)]},
        1: {'name': 'Agent1', 'configs': [(12456, 1), (16123, 2)]},
        3: {'name': 'Agent3', 'configs': [(13890, 0), (17234, 3)]},
        4: {'name': 'Agent4', 'configs': [(11567, 2), (19890, 1), (14890, 0)]}
    },
    3: {
        0: {'name': 'Agent0', 'configs': [(10123, 3), (15567, 1)]},
        1: {'name': 'Agent1', 'configs': [(12890, 2), (16567, 0), (19234, 3)]},
        2: {'name': 'Agent2', 'configs': [(14234, 1), (17890, 2)]},
        5: {'name': 'Agent5', 'configs': [(11890, 0), (18890, 3)]}
    },
    4: {
        0: {'name': 'Agent0', 'configs': [(10456, 2), (15123, 0), (19456, 1)]},
        1: {'name': 'Agent1', 'configs': [(13234, 3), (17123, 1)]},
        2: {'name': 'Agent2', 'configs': [(11123, 0), (16456, 2)]},
        3: {'name': 'Agent3', 'configs': [(14567, 1), (18123, 3), (19789, 0)]}
    },
    5: {
        0: {'name': 'Agent0', 'configs': [(10789, 1), (15456, 3)]},
        1: {'name': 'Agent1', 'configs': [(12123, 0), (16789, 2), (19012, 1)]},
        2: {'name': 'Agent2', 'configs': [(13567, 3), (17567, 0)]},
        4: {'name': 'Agent4', 'configs': [(11456, 2), (18456, 1)]}
    },
}
# C:\Users\matan\master_thesis\rl_envs\procgen\models\fruitbot\20251130-001800_easy
# sub_models_distance = {
#     0: [(1, 'FruitbotBase1', [])],
#     1: [(0, 'FruitbotEasy1', [])],
#     2: [(0, 'FruitbotEasy1', [])],
# }

# Action mappings for Fruitbot
actions_dict = {
    -1: "-1",
    1: "pass",
    0: "left",
    2: "right",
    3: "throw",
}

action_dir = {
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "Space": "throw",
    "ArrowDown": "pass",
}


# ------------------ UTILITY FUNCTION -----------------------------
# async def finish_turn(
#     response: Dict[str, Any],
#     user_game: GameControl,
#     sid: str,
#     need_feedback_data: bool = True
# ) -> None:
#     """Common logic after an action is processed."""
#     if response["done"]:
#         summary = user_game.end_of_episode_summary(need_feedback_data)
        
#         # Create video from episode frames
#         video_path = user_game.create_episode_video()
#         if video_path:
#             summary['video_path'] = video_path
        
#         # Send the summary to the front-end:
#         await sio.emit("episode_finished", summary, to=sid)
#     else:
#         await sio.emit("game_update", response, to=sid)

# -------------------- FASTAPI ROUTES ----------------------------
templates = Jinja2Templates(directory="templates")

@app.get("/health")
def health_check():
    """Health check endpoint for Docker and monitoring."""
    return {"status": "healthy", "service": "fruitbot-app"}

@app.get("/")
def index(request: Request) -> HTMLResponse:
    """
    Return index.html or a basic HTML if you don't have Jinja2 templates.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/update_action")
def update_action(payload: dict) -> Dict[str, Any]:
    index = payload["index"]
    action = payload["action"]
    return {"status": "action updated", "index_example": index, "action": action}

# -------------------- SOCKET.IO EVENTS ---------------------------
@sio.event
async def connect(sid: str, environ: Dict[str, Any]) -> None:
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid: str) -> None:
    print(f"Client disconnected: {sid}")
    # Remove the sid mapping (but keep the game control instance for future reconnects)
    if sid in sid_to_user:
        del sid_to_user[sid]

# def make_env_list():
#     env_kwargs = {
#             'food_diversity': 4,               # Variety of fruit sprites
#             'fruitbot_num_walls': 5,
#             'fruitbot_num_good_min': 10,
#             'fruitbot_num_good_range': 0,
#             'fruitbot_num_bad_min': 0,
#             'fruitbot_num_bad_range': 0,
#             'fruitbot_wall_gap_pct': 70,
#             'fruitbot_door_prob_pct': 0,
#             "use_discrete_action_wrapper": True,
#             "use_stay_bonus_wrapper": False
#         }

#         # Create Procgen Fruitbot environment
#     env_instance = gym.make(
#         'procgen:procgen-fruitbot-v0',
#         render_mode='rgb_array',
#         distribution_mode='easy',  # or 'hard', 'exploration', 'memory'
#         **env_kwargs
#     )
#     return [env_instance]
        

@sio.on("start_game")
async def start_game(sid: str, data: Dict[str, Any], callback: Optional[callable] = None) -> None:
    """
    When a user starts the game, they send their identifier (playerName).
    Create (or re-use) the GameControl instance corresponding to that user.
    """
    try:
        user_id = data.get("playerName", "")
        
        if not user_id:
            user_id = f"user_{sid[:8]}"
        
        sid_to_user[sid] = user_id
        
        if user_id not in game_controls:
            # Safely convert group to int, default to 0 if invalid
            try:
                group_val = data.get("group", "0")
                if isinstance(group_val, str) and len(group_val.strip()) > 1:
                    print(f"[start_game] Invalid group value '{group_val}', defaulting to 0")
                    similarity_level = 0
                else:
                    similarity_level = int(group_val)
            except (ValueError, TypeError):
                similarity_level = 0

            try:
                new_game = GameControl(
                    models_dict,
                    models_distance,
                    user_id,
                    similar_level_env=similarity_level,
                    feedback_partial_view=True,
                    env_seed_list=list(range(10000, 20000)),
                )
            except Exception as gc_error:
                print(f"[start_game] ERROR creating GameControl: {gc_error}")
                import traceback
                traceback.print_exc()
                raise
            
            game_controls[user_id] = new_game
            print(f"Created new game control for user {user_id} with similarity level {similarity_level}")

            # save a new user entry to the database
            if save_to_db:
                try:
                    session = SessionLocal()
                    new_user = Users(
                        user_id=user_id,
                        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        similarity_level=similarity_level,
                    )
                    session.add(new_user)
                    session.commit()
                    print(f"[start_game] New user {user_id} added to database")
                except Exception as db_error:
                    session.rollback()
                    print(f"[start_game] ERROR adding user to database: {db_error}")
                finally:
                    session.close()
        else:
            new_game = game_controls[user_id]
            print(f"Reusing existing game control for user {user_id}")
        
        if data.get("updateAgent", False):
            new_game.update_agent(data, sid)
            
        # if data.get("userNoFeedback", False):
        #     new_game.save_no_user_feedback(data, sid)
        
        response = new_game.get_initial_observation()
        response['action'] = None
        
        await sio.emit("game_update", response, to=sid)
        print(f"Game started successfully for user {user_id}")
        
    except Exception as e:
        print(f"[start_game] FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            await sio.emit("error", {
                "error": str(e), 
                "message": "Failed to start game. Please refresh and try again."
            }, to=sid)
        except Exception as emit_error:
            print(f"[start_game] Failed to send error to client: {emit_error}")

@sio.on("send_action")
async def handle_send_action(sid: str, action: str) -> Dict[str, str]:
    """
    Handle a user action. Look up the GameControl instance using the sid mapping.
    """
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
        
    user_game = game_controls[user_id]
    response = user_game.handle_action(action)
    response["action"] = action_dir.get(action, "Unknown")

    # if save_to_db:
    #     session = SessionLocal()
    #     try:
    #         obs_str = json.dumps(user_game.current_obs.tolist() if isinstance(user_game.current_obs, numpy.ndarray) else str(user_game.current_obs))[:1000]
    #         new_action = Action(
    #             action_type=str(action),
    #             agent_action=response["agent_action"],
    #             score=response["score"],
    #             reward=response["reward"],
    #             done=response["done"],
    #             user_id=user_game.user_id,
    #             timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    #             episode=response["episode"],
    #             env_state=obs_str
    #         )
    #         session.add(new_action)
    #         session.commit()
    #     except Exception as e:
    #         session.rollback()
    #         print(f"Database operation failed: {e}")
    #     finally:
    #         session.close()

    await finish_turn(response, user_game, sid, need_feedback_data=False)
    return {"status": "success"}

@sio.on("next_episode")
async def next_episode(sid: str, data: Optional[Dict[str, Any]] = None) -> None:
    # Accept optional data to match Socket.IO which may pass a payload (even if empty)
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.get_initial_observation()
    await sio.emit("game_update", response, to=sid)

@sio.on("ppo_action")
async def ppo_action(sid: str) -> None:
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.agent_action()
    await finish_turn(response, user_game, sid)

@sio.on("play_entire_episode")
async def play_entire_episode(sid: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Run the agent for a complete episode and stream frames in batches."""
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    
    user_game = game_controls[user_id]
    
    # Reset episode data
    episode_images = []
    episode_actions = []
    episode_rewards = []
    episode_positions = []  # Track bot x-positions
    total_score = 0
    step_count = 0
    
    # Bot tracking
    cx = 230  # Starting x position
    cx = dpu_clf.find_x_on_row(user_game.episode_frames[0]) + 15
    move_size = 35  # Movement amount per action
    
    # Streaming configuration
    BATCH_SIZE = 20  # Send frames every 20 steps
    batch_start_index = 0
    
    # Run the agent until episode is done
    done = False
    while not done and step_count < 1000:  # Safety limit
        # Get action from agent
        action, _ = user_game.ppo_agent.predict(user_game.current_obs, deterministic=True)
        action = action.item() if hasattr(action, 'item') else int(action)
        # if action == 3:
        #     action = 1  # Replace THROW with NOOP
        
        # Step environment
        result = user_game.step(action)
        done = result['done']

        # take only part of the observation (up action do not change much)
        if action == 1 and result['step_count'] % 2 > 0 and result['reward'] == 0 and not done:
            continue
        
        episode_images.append(result['image'])
        episode_actions.append(action)
        episode_rewards.append(float(result['reward']))
        total_score = result['score']
        step_count = result['step_count']
        
        # Update bot position based on action
        if action == 0:  # LEFT
            cx -= move_size
        elif action == 2:  # RIGHT
            cx += move_size
        # For other actions (1=UP/NOOP, 3=THROW), position stays same
        if not done:
            episode_positions.append(cx)
        
        # Stream batch every BATCH_SIZE steps
        if len(episode_images) - batch_start_index >= BATCH_SIZE or done:
            batch_data = {
                'images': episode_images[batch_start_index:],
                'actions': episode_actions[batch_start_index:],
                'rewards': episode_rewards[batch_start_index:],
                'positions': episode_positions[batch_start_index:],
                'collisions': {}, #user_game.collision_positions,  # Add collision data
                'score': float(total_score),  # Ensure native Python float
                'steps': int(step_count),      # Ensure native Python int
                'is_final': bool(done),        # Convert numpy bool_ to Python bool
                'batch_start': int(batch_start_index)
            }
            await sio.emit("episode_batch", batch_data, to=sid)
            batch_start_index = len(episode_images)
    
    # Send final episode_data event for compatibility (frontend can use batches or this)
    episode_data = {
        'images': episode_images,
        'actions': episode_actions,
        'rewards': episode_rewards,
        'positions': episode_positions,
        'collisions': {}, #user_game.collision_positions,  # Add collision data
        'score': float(total_score),  # Ensure native Python types for JSON
        'steps': int(step_count)
    }
    
    await sio.emit("episode_data", episode_data, to=sid)


@sio.on("play_episode")
async def play_episode(sid: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Alias for play_entire_episode - triggered by Play Agent button."""
    await play_entire_episode(sid, data)

@sio.on("compare_agents")
async def compare_agents(sid: str, data: Dict[str, Any]) -> None: # data={ playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions, similarity_level: similarity_level }  
    print(f"\n{'='*80}")
    print(f"[compare_agents] SOCKET EVENT RECEIVED")
    print(f"[compare_agents] SID: {sid}")
    print(f"[compare_agents] Data keys: {list(data.keys())}")
    print(f"[compare_agents] Update agent flag: {data.get('updateAgent')}")
    print(f"[compare_agents] User feedback count: {len(data.get('userFeedback', []))}")
    print(f"{'='*80}\n")
    
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        print(f"[compare_agents] ERROR: User not found - SID: {sid}, User ID: {user_id}")
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    
    user_game = game_controls[user_id]
    print(f"[compare_agents] User game found for: {user_id}")
    
    print(f"[compare_agents] Calling update_agent...")
    res = user_game.update_agent(data, sid)
    
    if res is None:
        print(f"[compare_agents] update_agent returned None, starting next episode")
        await next_episode(sid)
        return
        
    if user_game.similar_level_env == 0:
        current_path = user_game.current_agent_path
        print(f"[compare_agents] Similarity level is 0 - no visual comparison needed")
        print(f"  Current agent path: {current_path}")
        print(f"  Agent index: {user_game.agent_index}")
        await sio.emit("update_agent_group", {'agent_group': user_game.agent_index}, to=sid)
        if save_to_db:
            user_game.save_user_choice(True, '', datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[compare_agents] Update complete for user {user_id}")
        
        # For similarity level 0, send confirmation response instead of starting next episode
        print(f"[compare_agents] Emitting agent_updated_no_comparison response to client...")
        await sio.emit("compare_agents", {
            "similarity_level": 0,
            "agent_updated": True,
            "message": "Agent updated successfully"
        }, to=sid)
        print(f"[compare_agents] Response sent successfully")
        return
    
    print(f"[compare_agents] Calling agents_different_routs...")
    res = user_game.agents_different_routs(user_game.similar_level_env)
    
    if res and 'rawImageUpdated' in res and 'rawImagePrev' in res:
        print(f"[compare_agents] Successfully generated comparison images")
        print(f"  Image 1 size: {len(res['rawImageUpdated'])} chars")
        print(f"  Image 2 size: {len(res['rawImagePrev'])} chars")
        print(f"[compare_agents] Emitting compare_agents response to client...")
        await sio.emit("compare_agents", res, to=sid)
        print(f"[compare_agents] Response sent successfully")
    else:
        print(f"[compare_agents] ERROR: Failed to generate comparison images")
        print(f"  Response keys: {list(res.keys()) if res else 'None'}")
        await sio.emit("error", {"error": "Failed to generate comparison"}, to=sid)
    
    print(f"{'='*80}\n")

@sio.on("finish_game")
async def finish_game(sid: str) -> None:
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    scores = user_game.scores_lst
    await sio.emit("finish_game", {"scores": scores}, to=sid)

@sio.on("start_cover_page")
async def start_cover_page(sid: str) -> None:
    """
    Handle the transition from the cover page to the welcome page.
    """
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return

    # Call the start_game function in the background
    await start_game(sid, {"playerName": user_id})

    # Emit an event to transition to the welcome page
    await sio.emit("go_to_welcome_page", {}, to=sid)

@sio.on("agent_select")
async def agent_select(sid: str, data: Dict[str, Any]) -> None:
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return

    user_game = game_controls[user_id]

    demonstration_time_str = data.get('demonstration_time', None)
    if demonstration_time_str:
        try:
            dt = datetime.fromisoformat(demonstration_time_str.replace('Z', '+00:00'))
            demonstration_time_fmt = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Failed to parse demonstration_time: {demonstration_time_str}, error: {e}")
            demonstration_time_fmt = demonstration_time_str
    else:
        demonstration_time_fmt = datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    if save_to_db:
        user_game.save_user_choice(data['use_updated'], data.get('choiceExplanation', ''), demonstration_time_fmt)
        
    if data['use_updated'] == False:
        user_game.revert_to_old_agent()
        print(f"User {user_id} switched to the old agent.")
    else:
        print(f"User {user_id} keep with the new agent.")
    
    await sio.emit("agent_selection_result", {'agent_group': user_game.agent_index}, to=sid)

# ---------------------- RUNNING THE APP -------------------------
if __name__ == "__main__":
    if save_to_db:
        # clear_database()
        create_database()

    import uvicorn
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="warning",
        access_log=False
    )
