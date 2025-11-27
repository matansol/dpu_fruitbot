import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import io
import base64
from PIL import Image
import time
from functools import wraps
from stable_baselines3 import PPO
import os

def timeit(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def load_agent(env, model_path):
    """Load a PPO agent from a saved model."""
    from stable_baselines3 import PPO
    
    # Remove .zip extension if present since PPO.load() adds it automatically
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]
    
    # Verify file exists (check with .zip extension)
    zip_path = f"{model_path}.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model file not found: {zip_path}")
    
    try:
        model = PPO.load(model_path, env=env)
        print(f"Successfully loaded model from {zip_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {zip_path}: {e}")
        raise

def image_to_base64(img):
    """Convert numpy array image to base64 string for web display."""
    if img is None:
        raise ValueError("Image is None")
    
    # Procgen renders as RGB array (H, W, 3)
    if isinstance(img, np.ndarray):
        # Convert to PIL Image
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    else:
        raise TypeError(f"Expected numpy array, got {type(img)}")

@timeit
def capture_agent_path(env, agent, max_steps=1000):
    """
    Capture the path an agent takes in the environment.
    Returns: (move_sequence, images, final_score, actions)
    """
    obs, info = env.reset()
    done = False
    steps = 0
    total_reward = 0
    actions = []
    images = []
    move_sequence = []
    
    while not done and steps < max_steps:
        # Get agent action
        action, _ = agent.predict(obs, deterministic=True)
        action = action.item() if hasattr(action, 'item') else int(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        actions.append(action)
        
        # Render and store image
        img = env.render()
        if img is not None:
            images.append(img)
        
        # Store move for sequence (action number as move description)
        move_sequence.append((action, f"action_{action}"))
        
        steps += 1
    
    return move_sequence, images, total_reward, actions

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate an agent over multiple episodes."""
    scores = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            action = action.item() if hasattr(action, 'item') else int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        scores.append(total_reward)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'all_scores': scores
    }

def is_illegal_move(action, prev_pos, current_pos):
    """
    Check if a move was illegal (agent didn't move when it should have).
    For Procgen, all actions are valid, so this always returns False.
    """
    # Procgen doesn't have illegal moves in the same sense as grid environments
    return False

def actions_cells_locations(move_sequence):
    """
    Convert move sequence to cell locations.
    For Procgen, this is not applicable as there's no discrete grid.
    Returns empty list for compatibility.
    """
    return []

class GridAdapter:
    """
    Adapter class for grid-based visualizations.
    Not used in Procgen but kept for compatibility.
    """
    def __init__(self, env):
        self.env = env
    
    def get_grid_size(self):
        """Procgen doesn't have a grid, return observation shape."""
        obs_space = self.env.observation_space
        if hasattr(obs_space, 'shape'):
            return obs_space.shape[:2]  # Height, Width
        return (64, 64)  # Default Procgen size
    
    def get_agent_position(self):
        """Procgen doesn't expose agent position."""
        return None

def will_it_stuck(agent, env):
    """
    Check if agent will get stuck (truncated).
    For Procgen, episodes naturally terminate, so returns False.
    """
    return False

# Visualization functions for episode analysis (optional for Procgen)

def plot_episode_trajectory(images, actions, rewards):
    """
    Plot episode trajectory with images, actions, and rewards.
    Useful for debugging and analysis.
    """
    if not images:
        return None
    
    num_steps = len(images)
    fig, axes = plt.subplots(1, min(num_steps, 10), figsize=(20, 3))
    
    if num_steps == 1:
        axes = [axes]
    
    for idx, (img, action, reward) in enumerate(zip(images[:10], actions[:10], rewards[:10])):
        if isinstance(img, np.ndarray):
            axes[idx].imshow(img)
        axes[idx].set_title(f"Step {idx}\nA:{action}\nR:{reward:.2f}", fontsize=8)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str

def plot_rewards_over_time(cumulative_rewards):
    """Plot cumulative rewards over episode steps."""
    if not cumulative_rewards:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cumulative_rewards, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Episode Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str

def create_action_heatmap(actions, num_actions=9):
    """
    Create a heatmap showing action distribution.
    Useful for analyzing agent behavior patterns.
    """
    if not actions:
        return None
    
    action_counts = np.zeros(num_actions)
    for action in actions:
        if 0 <= action < num_actions:
            action_counts[action] += 1
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(num_actions), action_counts, color='#06A77D')
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_actions))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str

# Action name mappings for Fruitbot
ACTION_NAMES = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT", 
    3: "UP",
    4: "DOWN",
    5: "DOWN-LEFT",
    6: "DOWN-RIGHT",
    7: "UP-LEFT",
    8: "UP-RIGHT",
}

def get_action_name(action_idx):
    """Get human-readable name for action index."""
    return ACTION_NAMES.get(action_idx, f"ACTION_{action_idx}")


## Fruitbot code
def add_loc_on_observation(obs: np.ndarray, x_loc:int, y_loc:int) -> np.ndarray:
    """
    Get an game observation an 512x512x3 nparray and x and y location (0-15).
    Add to the array a blue dot at the specified location.
    """
    obs_copy = obs.copy()
    cell_size = 32  # Each cell is 32x32 pixels in 512x512 image
    center_x = x_loc * cell_size + cell_size // 2
    center_y = y_loc * cell_size + cell_size // 2
    
    # Draw a blue dot (circle) at the specified location
    rr, cc = np.ogrid[:obs_copy.shape[0], :obs_copy.shape[1]]
    circle = (rr - center_y) ** 2 + (cc - center_x) ** 2 <= (cell_size // 6) ** 2
    obs_copy[circle] = [0, 0, 255]  # Blue color in RGB
    
    return obs_copy





