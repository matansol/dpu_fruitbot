import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import copy
import numpy
import json
from datetime import datetime
import random
import sys

# Add procgen imports
import gymnasium as gym
import procgen

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
from dpu_clf import (
    load_agent,
    capture_agent_path,
    evaluate_agent,
    is_illegal_move,
    actions_cells_locations,
    GridAdapter,
    image_to_base64,
    timeit,
)

from functools import reduce
from typing import Dict, Any

# Procgen Fruitbot action constants
# Procgen uses 15 discrete actions (standard for most Procgen games)
FRUITBOT_ACTIONS = {
    'LEFT': 1,
    'RIGHT': 2,
    'UP': 3,
    'DOWN': 4,
    'DOWNLEFT': 5,
    'DOWNRIGHT': 6,
    'UPLEFT': 7,
    'UPRIGHT': 8,
    'NOOP': 0,
    # Additional actions (9-14) are game-specific
}

# ---------------------- FRUITBOT HELPER FUNCTIONS ----------------------

def get_fruitbot_position(env):
    """Extract agent position from Procgen environment if available."""
    # Procgen doesn't expose position directly, return None
    # Position would need to be tracked through observations
    return None

def get_fruitbot_observation(env):
    """Get observation from Fruitbot environment."""
    obs, info = env.reset() if not hasattr(env, '_obs') else (env._obs, {})
    return obs

def will_it_stuck(agent, env):
    """Check if agent will get stuck - adapted for Procgen."""
    # Procgen has max_steps built-in, no custom truncation needed
    return False

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

# SQLAlchemy setup
DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")
engine = create_engine(DATABASE_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
# Global variable to control database saving

save_to_db = True

class Action(Base):
    __tablename__ = "actions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    action_type = Column(String(20))
    agent_action = Column(Boolean)
    score = Column(Float)
    reward = Column(Float)
    done = Column(Boolean)
    episode = Column(Integer)
    timestamp = Column(String(30))
    agent_index = Column(Integer)
    env_state = Column(String(1000))

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
    env_state = Column(String(1000))
    agent_action = Column(String(20))
    feedback_action = Column(String(20))
    feedback_explanation = Column(String(500), nullable=True)
    action_index = Column(Integer)
    timestamp = Column(String(30))  
    episode_index = Column(Integer)
    agent_path = Column(String(100))
    similarity_level = Column(Integer)
    feedback_unique_env = Column(Integer, nullable=True)

class UserChoice(Base): 
    __tablename__ = "user_choices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    old_agent_path = Column(String(100)) 
    new_agent_path = Column(String(100))
    old_agent_score_list = Column(String(30))
    new_agent_score_list = Column(String(30))
    timestamp = Column(String(50))
    demonstration_time = Column(String(50))
    episode_index = Column(Integer)
    choice_to_update = Column(Boolean)
    choice_explanation = Column(String(500), nullable=True)
    similarity_level = Column(Integer)
    feedback_score = Column(Float, nullable=True)
    feedback_count = Column(Integer, nullable=True)
    unique_envs = Column(String(20), nullable=True)
    examples_shown = Column(Integer, nullable=True)

def clear_database():
    """Clears the database tables."""
    print("Clearing database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def create_database():
    """Creates the database tables if they do not already exist."""
    print("Ensuring database tables are created...")
    Base.metadata.create_all(bind=engine)


# Helper
async def in_thread(func, *args, **kw):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kw))


class GameControl:
    def __init__(self, env, models_paths, models_distance, user_id, similar_level_env=0, feedback_partial_view=True):
        self.env = env
        self.agent_index = 0  # Start with first agent
        self.models_paths = models_paths
        self.models_distance = models_distance
        self.episode_num = 0
        self.scores_lst = []
        self.last_obs = None
        self.episode_actions = []
        self.episode_cumulative_rewards = []
        self.agent_last_pos = None
        self.episode_images = []
        self.episode_obs = []
        self.episode_agent_locations = []
        self.invalid_moves = 0
        self.user_id = user_id
        self.lava_penalty: float = -3.0
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
        # self.last_unique_env: int = 0 # the last unique env that was given to the user
        
    @timeit
    def reset(self):
        self.update_agent(None, None)
        
        # Procgen reset returns observation and info
        obs, info = self.env.reset()
        
        # Store observation (Procgen returns RGB array directly)
        self.episode_obs = [obs]
        
        # Procgen doesn't expose position, track as None
        self.episode_agent_locations = [(None, None)]
        
        self.feedback_score = 0
        self.saved_env = copy.deepcopy(self.env)
        
        self.saved_env_info = {
            'max_steps': 1000,  # Procgen default
        }

        self.score = 0
        self.invalid_moves = 0
        return obs

    @timeit
    def actions_to_moves_sequence(self, episode_actions):
        small_arrow = 'turn '  # small arrow is used to indicate the agent turning left or right
        agent_dir = "right"
        move_sequence = []
        for action in episode_actions:
            if action == 0:  # turn left
                agent_dir = turn_agent(agent_dir, "left")
                move_sequence.append((agent_dir, 'turn left'))
            elif action == 1:  # turn right
                agent_dir = turn_agent(agent_dir, "right")
                move_sequence.append((agent_dir, 'turn right'))
            elif action == 2:  # move forward
                move_sequence.append((agent_dir, 'forward'))
            elif action == 3:  # pickup
                move_sequence.append((agent_dir, 'pickup'))
            else:
                move_sequence.append(("invalide move", "invalide move"))
        return move_sequence

    # @timeit
    def step(self, action, agent_action=False):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.score += reward
        self.score = round(self.score, 1)
        if done:
            self.scores_lst.append(self.score)
            self.last_score = self.score
        
        # Procgen doesn't have illegal moves in the same way
        # All actions are valid, just add to episode
        self.episode_actions.append(action)
        if self.episode_cumulative_rewards:
            self.episode_cumulative_rewards.append(round(self.episode_cumulative_rewards[-1] + reward, 1))
        else:
            self.episode_cumulative_rewards.append(round(reward, 1))
        self.episode_obs.append(observation)
        self.episode_agent_locations.append((None, None))  # No position tracking
        
        # Procgen render returns RGB array
        img = self.env.render()
        image_base64 = image_to_base64(img)
        self.episode_images.append(image_base64)

        self.current_obs = observation
        
        # Get step count from info if available
        step_count = info.get('level_seed_step', 0)
        
        return {
            'image': image_base64,
            'episode': self.episode_num,
            'reward': reward,
            'done': done,
            'score': self.score,
            'last_score': self.last_score,
            'agent_action': agent_action,
            'step_count': step_count,
            'agent_index': self.agent_index
        }

    def handle_action(self, action_str):
        """Map keyboard input to Fruitbot actions."""
        key_to_action = {
            "ArrowLeft": FRUITBOT_ACTIONS['LEFT'],
            "ArrowRight": FRUITBOT_ACTIONS['RIGHT'],
            "ArrowUp": FRUITBOT_ACTIONS['UP'],
            "ArrowDown": FRUITBOT_ACTIONS['DOWN'],
            "Space": FRUITBOT_ACTIONS['NOOP'],
            "q": FRUITBOT_ACTIONS['UPLEFT'],
            "w": FRUITBOT_ACTIONS['UP'],
            "e": FRUITBOT_ACTIONS['UPRIGHT'],
            "a": FRUITBOT_ACTIONS['LEFT'],
            "d": FRUITBOT_ACTIONS['RIGHT'],
            "z": FRUITBOT_ACTIONS['DOWNLEFT'],
            "x": FRUITBOT_ACTIONS['DOWN'],
            "c": FRUITBOT_ACTIONS['DOWNRIGHT'],
        }
        action = key_to_action.get(action_str, FRUITBOT_ACTIONS['NOOP'])
        return self.step(action)

    @timeit
    def get_initial_observation(self):
    
        self.current_obs = self.reset()
        self.agent_last_pos = None
        self.episode_actions = []
        self.episode_cumulative_rewards = []
        
        # Render initial frame
        img = self.env.render()
        if img is None:
            raise Exception("initial observation rendering failed")
        image_base64 = image_to_base64(img)
        self.episode_images = [image_base64]  # Store base64 string, not raw image
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

    def agent_action(self):
        # Get action from PPO agent
        agent_config = self.models_paths[self.agent_index]
        
        # PPO agent: predict returns (action, _states)
        action, _ = self.ppo_agent.predict(self.current_obs, deterministic=True)
        action = action.item() if hasattr(action, 'item') else int(action)
        
        result = self.step(action, True)
        result['action'] = action
        return result

    # TODO: Pacman version needed
    # def update_env_to_action(self, action_index):
    #     tmp_env = copy.deepcopy(self.saved_env)
    #     obs = tmp_env.get_wrapper_attr('current_state')
    #     for action in self.episode_actions[:action_index]:
    #         obs, r, ter, tru, info = tmp_env.step(action)
    #     return tmp_env, obs

    def revert_to_old_agent(self):
        self.ppo_agent = self.prev_agent
        self.agent_index = self.prev_agent_index
        self.current_agent_path = self.models_paths[self.agent_index]['path']
        print(f'(revert) prev agent index={self.prev_agent_index}, prev agent path={self.prev_agent_path}')
        print(f"(revert) current agent index={self.agent_index}")

    def count_similar_actions(self, env, other_agent, other_agent_config, feedback_indexes):
        similar_actions = 0
        
        for i, action in enumerate(self.episode_actions):
            if i in feedback_indexes:
                continue
            saved_obs = self.episode_obs[i]
            
            # Get predicted action from other agent
            predicted_action = other_agent.predict(saved_obs, deterministic=True)[0]
            predicted_action = predicted_action.item() if hasattr(predicted_action, 'item') else int(predicted_action)
            
            if action == predicted_action:
                similar_actions += 1
        return similar_actions

    # TODO: Pacman version needed - MiniGrid-specific feedback evaluation
    # def is_good_feedback(self, base_front_object, agent_front_object, feedback_front_object):
    #     """Check if the feedback action was towards a good ball (blue or green) or the agent's action was thowards a bad object (red ball or lava)."""
    #     if base_front_object and (IDX_TO_OBJECT[base_front_object[0]] == 'lava' or (IDX_TO_OBJECT[base_front_object[0]] == 'ball' and IDX_TO_COLOR[base_front_object[1]] == 'red')):
    #         return True  # base object is bad, so any feedback is good
    #     if agent_front_object and (IDX_TO_OBJECT[agent_front_object[0]] == 'lava' or (IDX_TO_OBJECT[agent_front_object[0]] == 'ball' and IDX_TO_COLOR[agent_front_object[1]] == 'red')):
    #         return True  # agent action was bad, so any feedback is good
    #     if feedback_front_object and (IDX_TO_OBJECT[feedback_front_object[0]] == 'ball' and IDX_TO_COLOR[feedback_front_object[1]] in ['blue', 'green']):
    #         return True  # feedback action was good, so it's a good feedback
    #     return False

    @timeit
    def update_agent(self, data, sid):
        print(f"(update agent), data={data}")
        if self.ppo_agent is None:
            agent_config = self.models_paths[self.agent_index]
            self.ppo_agent = load_agent(self.env, agent_config['path'])
            self.current_agent_path = agent_config['path']
            self.prev_agent = self.ppo_agent
            self.prev_agent_path = self.current_agent_path
            self.prev_agent_index = self.agent_index
            print(f'Loaded the first model: {agent_config["name"]}')
            print(f'(update agent None) prev agent index={self.prev_agent_index}')
            print(f"(update agent None) current agent index={self.agent_index}")
            return None
        if data is None:
            print("Data is None, return")
            return None
        if data.get('updateAgent', False) == False:
            print("No need for update, return")
            return None
        user_feedback = data.get('userFeedback')
        if user_feedback is None or len(user_feedback) == 0:
            print("No user feedback, return")
            return None
        
        # Remove duplicates
        unique_feedback = {}
        for feedback in user_feedback:
            unique_feedback[feedback['index']] = feedback
        user_feedback = list(unique_feedback.values())
        self.number_of_feedbacks = len(user_feedback)

        # Save feedback to DB
        if save_to_db and sid:
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
                        env_state=obs_str[:1000],  # Truncate to fit column
                        agent_action=str(agent_action),
                        feedback_action=str(action_feedback['feedback_action']),
                        feedback_explanation=action_feedback.get('feedback_explanation', ''),
                        action_index=action_index,
                        episode_index=self.episode_num,
                        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        agent_path=self.current_agent_path,
                        similarity_level=self.similar_level_env,
                        feedback_unique_env=self.board_seen[-1] if self.board_seen else 0
                    )
                    session.add(feedback_action)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Database operation failed: {e}")
                finally:
                    session.close()

        feedback_indexes = [feedback['index'] for feedback in user_feedback]

        optimal_agents = []
        target_models_indexes = self.models_distance[self.agent_index]
        for model_i, model_name, _ in target_models_indexes:
            agent_data = self.models_paths[model_i]
            path = agent_data['path']
            agent = load_agent(self.env, path)

            agent_correctness = 0
            for action_feedback in user_feedback:
                if action_feedback['index'] >= len(self.episode_obs):
                    return
                saved_obs = self.episode_obs[action_feedback['index']]
                
                agent_predict_action = agent.predict(saved_obs, deterministic=True)
                agent_predict_action = int(agent_predict_action[0].item() if hasattr(agent_predict_action[0], 'item') else agent_predict_action[0])
                
                if agent_predict_action == int(action_feedback['feedback_action']):
                    agent_correctness += 1

            if agent_correctness > 0:
                similar_actions = self.count_similar_actions(copy.deepcopy(self.saved_env), agent, agent_data, feedback_indexes)
                optimal_agents.append({
                    "agent": agent,
                    "name": model_name,
                    "path": path,
                    "correctness_feedback": agent_correctness,
                    "similar_actions": similar_actions,
                    "model_index": model_i
                })

        print(f"{'_'*50} counted the feedback - optimal_agents:")
        for agent_dict in optimal_agents:
            print(f"{agent_dict['name']} correctness_feedback={agent_dict['correctness_feedback']}, similar_actions={agent_dict['similar_actions']}")

        if len(optimal_agents) == 0:
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
        print(f"User_id={self.user_id}, current agent is {self.agent_index}: {self.models_paths[self.agent_index]['name']}")
        print(f"User_id={self.user_id}, new agent picked is:{new_agent_dict['name']}")
        self.prev_agent = self.ppo_agent
        self.prev_agent_path = self.current_agent_path
        self.prev_agent_index = self.agent_index
        self.ppo_agent = new_agent_dict["agent"]
        self.agent_index = new_agent_dict["model_index"]
        print(f'(update agent) prev agent index={self.prev_agent_index}, prev agent path={self.prev_agent_path}')
        print(f"(update_agent) current agent index={self.agent_index}")
        self.current_agent_path = self.models_paths[self.agent_index]['path']
        if self.prev_agent is None:
            self.prev_agent = self.ppo_agent
        return True

    @timeit
    def agents_different_routs(self, similarity_level=5, stuck_count=0, same_path_count=0):
        if self.ppo_agent == None or self.prev_agent == None:
            print(f"No two agents to compare ppo_agent: {self.ppo_agent}, prev_agent: {self.prev_agent}")
            if self.ppo_agent == None and self.prev_agent == None:
                self.ppo_agent = self.prev_agent = load_agent(self.env, self.models_paths[self.agent_index]['path'])
            if self.ppo_agent == None:
                self.ppo_agent = self.prev_agent
                self.agent_index = self.prev_agent_index
                self.current_agent_path = self.models_paths[self.agent_index]['path']
            else:
                self.prev_agent_index = 0
                self.prev_agent_path = self.models_paths[self.prev_agent_index]['path']
                self.prev_agent = load_agent(self.env, self.prev_agent_path)

        env = self.saved_env
                
        copy_env = copy.deepcopy(env)
        img = copy_env.render()
        updated_move_sequence, _, current_score, agent_actions = capture_agent_path(copy_env, self.ppo_agent)
        self.current_agent_score_list.append(current_score)

        copy_env = copy.deepcopy(env)
        prev_move_sequence, _, prev_score, prev_agent_actions = capture_agent_path(copy_env, self.prev_agent)
        self.prev_agent_score_list.append(prev_score)
        
        if (prev_move_sequence == updated_move_sequence) and same_path_count < 3:
            print(f"User_id={self.user_id}, agents_different_routs {same_path_count} times, trying again")
            return self.agents_different_routs(similarity_level=similarity_level, same_path_count=same_path_count+1)
        
        self.past_choices.add((self.models_paths[self.prev_agent_index]['name'], self.models_paths[self.agent_index]['name']))
        converge_action_index = -1
        image_base64 = image_to_base64(img)
        
        return {
            'rawImage': image_base64,
            'prevMoveSequence': [],
            'updatedMoveSequence': [],
            'converge_action_index': converge_action_index
        }

    @timeit
    def end_of_episode_summary(self, need_feedback_data:bool = True):
        path_img_base64 = None
        actions_locations = []
        images_buf_list = [] if need_feedback_data else None
        actions_cells = [] if need_feedback_data else None

        return {
            'path_image': path_img_base64,
            'actions': actions_locations,
            'cumulative_rewards': self.episode_cumulative_rewards,
            'invalid_moves': self.invalid_moves,
            'score': self.last_score,
            'feedback_images': self.episode_images,
            'actions_cells': actions_cells,
            'feedback_score': self.feedback_score - (self.number_of_feedbacks - self.feedback_score),
        }

    def save_no_user_feedback(self, data, sid):
        user_explanation = data.get('userExplanation')
        if save_to_db and sid:
            try:
                session = SessionLocal()
                action_index = 0
                if action_index >= len(self.episode_obs):
                    obs_str = "No observation available"
                else:
                    obs = self.episode_obs[action_index]
                    obs_str = json.dumps(obs.tolist() if isinstance(obs, numpy.ndarray) else str(obs))[:1000]
                
                feedback_action = FeedbackAction(
                    user_id=self.user_id,
                    env_state=obs_str,
                    agent_action="-1",
                    feedback_action="-1",
                    feedback_explanation=user_explanation,
                    action_index=-1,
                    episode_index=self.episode_num,
                    timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    agent_path=self.current_agent_path,
                    similarity_level=self.similar_level_env,
                    feedback_unique_env=self.board_seen[-1] if self.board_seen else 0
                )
                session.add(feedback_action)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Database operation failed: {e}")
            finally:
                session.close()

    def save_user_choice(self, choice_to_update, choice_explanation, demonstration_time_fmt):
        session = SessionLocal()
        try:
            unique_env = self.demonstration_unique_envs[-1] if self.demonstration_unique_envs else (self.board_seen[-1] if self.board_seen else 0)
            user_choice = UserChoice(
                user_id=self.user_id,
                old_agent_path=str(self.models_paths[self.prev_agent_index]['path']),
                new_agent_path=str(self.current_agent_path),
                old_agent_score_list=','.join(str(x) for x in self.prev_agent_score_list),
                new_agent_score_list=','.join(str(x) for x in self.current_agent_score_list),
                timestamp=datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"),
                demonstration_time=demonstration_time_fmt,
                episode_index=self.episode_num,
                choice_to_update=choice_to_update,
                choice_explanation=choice_explanation,
                similarity_level=self.similar_level_env,
                feedback_score=self.feedback_score,
                feedback_count=self.number_of_feedbacks,
                unique_envs=str(unique_env),
                examples_shown=1,
            )
            session.add(user_choice)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"UserChoice saving failed: {e}")
        finally:
            session.close()

    def revert_to_old_agent(self):
        self.ppo_agent = self.prev_agent
        self.agent_index = self.prev_agent_index
        self.current_agent_path = self.models_paths[self.agent_index]['path']
        print(f'(revert) prev agent index={self.prev_agent_index}, prev agent path={self.prev_agent_path}')
        print(f"(revert) current agent index={self.agent_index}")

# ---------------- Global Variables ----------------

game_controls: Dict[str, GameControl] = {}
sid_to_user: Dict[str, str] = {}

# Procgen Fruitbot models configuration
# TODO: Replace these paths with your actual trained Fruitbot models
sub_models_dict = {
    0: {'path': 'models/fruitbot/20251117-173536_easy/ppo_final.zip', 'name': 'FruitbotEasy1', 'type': 'ppo', 'vector': (1, 1, 1, 1, 1)},
    1: {'path': 'models/fruitbot/20251116-195636/ppo_final.zip', 'name': 'FruitbotBase1', 'type': 'ppo', 'vector': (2, 2, 2, 2, 2)},
    # Add more trained agents as needed
}

sub_models_distance = {
    0: [(1, 'FruitbotBase1', [])],
    1: [(0, 'FruitbotEasy1', [])],
}


# Action mappings for Fruitbot
actions_dict = {
    -1: "-1",
    0: "pass",
    1: "left",
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
async def finish_turn(response: dict, user_game: GameControl, sid: str, need_feedback_data: bool = True):
    """Common logic after an action is processed."""
    if response["done"]:
        summary = user_game.end_of_episode_summary(need_feedback_data)
        # Send the summary to the front-end:
        await sio.emit("episode_finished", summary, to=sid)
    else:
        await sio.emit("game_update", response, to=sid)

# -------------------- FASTAPI ROUTES ----------------------------
templates = Jinja2Templates(directory="templates")

@app.get("/")
def index(request: Request):
    """
    Return index.html or a basic HTML if you don't have Jinja2 templates.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/update_action")
def update_action(payload: dict):
    index = payload["index"]
    action = payload["action"]
    return {"status": "action updated", "index_example": index, "action": action}

# -------------------- SOCKET.IO EVENTS ---------------------------
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # Remove the sid mapping (but keep the game control instance for future reconnects)
    if sid in sid_to_user:
        del sid_to_user[sid]

@sio.on("start_game")
async def start_game(sid, data, callback=None):
    """
    When a user starts the game, they send their identifier (playerName).
    Create (or re-use) the GameControl instance corresponding to that user.
    """
    print("starting the game")
    user_id = data["playerName"]
    
    sid_to_user[sid] = user_id
    if user_id not in game_controls:
        # Safely convert group to int, default to 1 if invalid
        try:
            group_val = data.get("group", "0")
            # Check if it's a template variable or other invalid value
            if isinstance(group_val, str) and "${" in group_val:
                similarity_level = 0
            else:
                similarity_level = int(group_val)
        except (ValueError, TypeError):
            similarity_level = 0
            print(f"(user_ID={user_id})  Invalid group value: {data.get('group')}, defaulting to 0")

        # Create Procgen Fruitbot environment
        env_instance = gym.make(
            'procgen:procgen-fruitbot-v0',
            render_mode='rgb_array',
            distribution_mode='easy'  # or 'hard', 'exploration', 'memory'
        )
        
        new_game = GameControl(
            env_instance,
            sub_models_dict,
            sub_models_distance,
            user_id,
            similar_level_env=similarity_level,
            feedback_partial_view=True
        )
        game_controls[user_id] = new_game
        print(f"Created new game control for user {user_id} with similarity level {similarity_level}")
    else:
        new_game = game_controls[user_id]
        print(f"Reusing existing game control for user {user_id}")
    
    if data.get("updateAgent", False):
        new_game.update_agent(data, sid)
    if data.get("userNoFeedback", False):
        new_game.save_no_user_feedback(data, sid)
    
    response = new_game.get_initial_observation()
    response['action'] = None
    await sio.emit("game_update", response, to=sid)

@sio.on("send_action")
async def handle_send_action(sid, action):
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

    if save_to_db:
        session = SessionLocal()
        try:
            obs_str = json.dumps(user_game.current_obs.tolist() if isinstance(user_game.current_obs, numpy.ndarray) else str(user_game.current_obs))[:1000]
            new_action = Action(
                action_type=str(action),
                agent_action=response["agent_action"],
                score=response["score"],
                reward=response["reward"],
                done=response["done"],
                user_id=user_game.user_id,
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                episode=response["episode"],
                env_state=obs_str
            )
            session.add(new_action)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Database operation failed: {e}")
        finally:
            session.close()

    await finish_turn(response, user_game, sid, need_feedback_data=False)
    return {"status": "success"}

@sio.on("next_episode")
async def next_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.get_initial_observation()
    await sio.emit("game_update", response, to=sid)

@sio.on("ppo_action")
async def ppo_action(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    response = user_game.agent_action()
    await finish_turn(response, user_game, sid)

@sio.on("play_entire_episode")
async def play_entire_episode(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    while True:
        response = user_game.agent_action()
        await asyncio.sleep(0.3)
        await finish_turn(response, user_game, sid)
        if response["done"]:
            await asyncio.sleep(0.1)
            break
    user_game.prev_agent_score_list = []
    user_game.current_agent_score_list = []


@sio.on("compare_agents")
async def compare_agents(sid, data): # data={ playerName: playerNameInput.value, updateAgent: true, userFeedback: userFeedback, actions: actions, similarity_level: similarity_level }  
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    
    res = user_game.update_agent(data, sid)
    if res is None:
        await next_episode(sid)
        return
    if user_game.similar_level_env == 0:
        current_path = user_game.current_agent_path
        print(f"Current agent path: {current_path}")
        print(f"User has updated the agent, similarity level is 0, agent idx = {user_game.agent_index}")
        await sio.emit("update_agent_group", {'agent_group': user_game.agent_index}, to=sid)
        user_game.save_user_choice(1, '', datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S"))
        print(f"User {user_id} updated the agent, similarity level is 0, no comparison")
        return
    
    res = user_game.agents_different_routs(user_game.similar_level_env)
    await sio.emit("compare_agents", res, to=sid)

@sio.on("finish_game")
async def finish_game(sid):
    user_id = sid_to_user.get(sid)
    if not user_id or user_id not in game_controls:
        await sio.emit("error", {"error": "User not found"}, to=sid)
        return
    user_game = game_controls[user_id]
    scores = user_game.scores_lst
    await sio.emit("finish_game", {"scores": scores}, to=sid)

@sio.on("start_cover_page")
async def start_cover_page(sid):
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

@sio.on("agent_selected")
async def agent_selected(sid, data):
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
        user_game.save_user_choice(not data['use_old_agent'], data.get('choiceExplanation', ''), demonstration_time_fmt)
        
    if data['use_old_agent']:
        user_game.revert_to_old_agent()
        print(f"User {user_id} switched to the old agent.")
    else:
        print(f"User {user_id} keep with the new agent.")
    
    await sio.emit("agent_selection_result", {'agent_group': user_game.agent_index}, to=sid)

# ---------------------- RUNNING THE APP -------------------------
if __name__ == "__main__":
    save_to_db = False
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
