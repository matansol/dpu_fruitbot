import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import io
import base64
from PIL import Image, ImageDraw
import time
from functools import wraps
from stable_baselines3 import PPO
import os

import cv2
from IPython.display import display

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
    0: "Left",
    1: "Stay",
    2: "RIGHT", 
    3: "Throw",
}

def get_action_name(action_idx):
    """Get human-readable name for action index."""
    return ACTION_NAMES.get(action_idx, f"ACTION_{action_idx}")


## Fruitbot code

#find the fruitbot color in rgb
def analyze_fruitbot_colors(img_path: str):
    """
    Analyze the colors of the fruitbot in a sample frame.
    Identify gray/black/white colors and detect the bot's position.
    """
    # Load test frame and resize to 512x512x3
    img_path = "tests/frameshots/fruitbot_frame_12.png"
    img = Image.open(img_path)
    img = img.resize((512, 512))
    frame = np.array(img)

    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    # Focus on bottom 1/5 of frame where bot is located
    h, w, _ = frame.shape
    y_start = int(h * 0.9)
    bottom_region = frame[y_start:y_start+2, :, :]



    print(f"Bottom region shape: {bottom_region.shape} (y from {y_start} to {h})")
    display(Image.fromarray(bottom_region))

    # Analyze colors in bottom region
    pixels = bottom_region.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Filter for GRAY, BLACK, and WHITE colors only (bot-like colors)
    # Gray: low variance between R,G,B channels, mid-range values
    # Black: all channels low
    # White: all channels high
    def is_gray_black_or_white(rgb, max_variance=30, gray_min=60, gray_max=200, black_max=60, white_min=200):
        """Check if RGB color is gray-ish, black, or white"""
        r, g, b = rgb
        variance = np.var([r, g, b])
        mean_val = np.mean([r, g, b])
        
        is_black = (r < black_max) and (g < black_max) and (b < black_max)
        is_white = (r >= white_min) and (g >= white_min) and (b >= white_min) and (variance < max_variance)
        is_gray = (variance < max_variance) and (gray_min <= mean_val <= gray_max)
        
        return is_black or is_gray or is_white

    # Filter to bot-like colors
    bot_color_mask = np.array([is_gray_black_or_white(col) for col in unique_colors])
    bot_colors = unique_colors[bot_color_mask]
    bot_counts = counts[bot_color_mask]

    print(f"\nFiltered to {len(bot_colors)} gray/black/white colors (from {len(unique_colors)} total)")

    if len(bot_colors) > 0:
        bot_order = np.argsort(-bot_counts)
        
        print("\nTop 30 GRAY/BLACK/WHITE colors (RGB) in bottom region:")
        print("Rank | RGB Color        | Count  | Fraction | Hex      | Category")
        print("-" * 85)
        for i, idx in enumerate(bot_order[:30], 1):
            col = tuple(int(v) for v in bot_colors[idx])
            cnt = int(bot_counts[idx])
            frac = cnt / float(pixels.shape[0])
            hex_color = "#{:02x}{:02x}{:02x}".format(*col)
            
            # Categorize
            if max(col) < 60:
                category = "BLACK"
            elif min(col) >= 200 and np.var(col) < 30:
                category = "WHITE"
            elif np.var(col) < 10:
                category = "GRAY (very uniform)"
            else:
                category = "GRAY (slight tint)"
            
            print(f"{i:4} | {col!s:16} | {cnt:6} | {frac:6.2%} | {hex_color} | {category}")
        
        # Create palette of top 20 gray/black/white colors
        top_n = min(20, len(bot_order))
        palette_h = 60
        palette = np.zeros((palette_h, 40 * top_n, 3), dtype=np.uint8)
        for i, idx in enumerate(bot_order[:top_n]):
            palette[:, i*40:(i+1)*40, :] = bot_colors[idx]
        
        print("\nColor palette (top 20 gray/black/white colors):")
        display(Image.fromarray(palette))
        
        # Now detect bot using refined thresholds from actual colors
        # Get the range of gray/black/white colors actually present
        if len(bot_colors) > 0:
            black_colors = bot_colors[[max(c) < 60 for c in bot_colors]]
            white_colors = bot_colors[[min(c) >= 200 and np.var(c) < 30 for c in bot_colors]]
            gray_colors = bot_colors[[not (max(c) < 60 or (min(c) >= 200 and np.var(c) < 30)) for c in bot_colors]]
            
            print(f"\nColor statistics from detected bot colors:")
            if len(black_colors) > 0:
                print(f"  Black colors: {len(black_colors)}")
                print(f"    Max value: {black_colors.max()}")
            if len(white_colors) > 0:
                print(f"  White colors: {len(white_colors)}")
                print(f"    Min value: {white_colors.min()}")
                print(f"    R range: [{white_colors[:,0].min()}, {white_colors[:,0].max()}]")
                print(f"    G range: [{white_colors[:,1].min()}, {white_colors[:,1].max()}]")
                print(f"    B range: [{white_colors[:,2].min()}, {white_colors[:,2].max()}]")
            if len(gray_colors) > 0:
                print(f"  Gray colors: {len(gray_colors)}")
                print(f"    R range: [{gray_colors[:,0].min()}, {gray_colors[:,0].max()}]")
                print(f"    G range: [{gray_colors[:,1].min()}, {gray_colors[:,1].max()}]")
                print(f"    B range: [{gray_colors[:,2].min()}, {gray_colors[:,2].max()}]")
    else:
        print("No gray/black/white colors found - bot may not be visible in this region")

    # Detect bot using gray/black/white color masks
    gray_mask = np.zeros(bottom_region.shape[:2], dtype=np.uint8)
    black_mask = np.zeros(bottom_region.shape[:2], dtype=np.uint8)
    white_mask = np.zeros(bottom_region.shape[:2], dtype=np.uint8)

    for y in range(bottom_region.shape[0]):
        for x in range(bottom_region.shape[1]):
            rgb = tuple(bottom_region[y, x, :])
            if is_gray_black_or_white(rgb):
                if max(rgb) < 60:
                    black_mask[y, x] = 255
                elif min(rgb) >= 200 and np.var(rgb) < 30:
                    white_mask[y, x] = 255
                else:
                    gray_mask[y, x] = 255

    combined_mask = cv2.bitwise_or(cv2.bitwise_or(gray_mask, black_mask), white_mask)

    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)

    print(f"\nDetected {num_labels - 1} connected components from gray/black/white pixels")
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w_box = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h_box = stats[largest_idx, cv2.CC_STAT_HEIGHT]
        area = stats[largest_idx, cv2.CC_STAT_AREA]
        cx, cy = centroids[largest_idx]
        
        print(f"\nLargest component (likely the bot):")
        print(f"  Bounding box: x={x}, y={y}, w={w_box}, h={h_box}")
        print(f"  Area: {area} pixels")
        print(f"  Centroid: ({cx:.1f}, {cy:.1f})")
        print(f"  Global position: x={x}, y={y_start + y}")
        
        # Visualize detection
        vis = bottom_region.copy()
        vis = frame.copy()
        # cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.circle(vis, (int(cx)+15, 470), 5, (255, 0, 0), -1)
        
        print("\nBot detection visualization:")
        display(Image.fromarray(vis))
        
        # Extract bot region and re-analyze its colors
        bot_region = bottom_region[y:y+h_box, x:x+w_box, :]
        bot_pixels_extracted = bot_region.reshape(-1, 3)
        bot_unique_extracted, bot_counts_extracted = np.unique(bot_pixels_extracted, axis=0, return_counts=True)
        
        # Filter to gray/black/white in extracted region
        bot_gbw_mask = np.array([is_gray_black_or_white(col) for col in bot_unique_extracted])
        bot_gbw_colors = bot_unique_extracted[bot_gbw_mask]
        bot_gbw_counts = bot_counts_extracted[bot_gbw_mask]
        
        if len(bot_gbw_colors) > 0:
            bot_gbw_order = np.argsort(-bot_gbw_counts)
            
            print(f"\nTop 10 GRAY/BLACK/WHITE colors in detected bot region:")
            print("Rank | RGB Color        | Count  | Fraction | Hex      | Category")
            print("-" * 80)
            for i, idx in enumerate(bot_gbw_order[:10], 1):
                col = tuple(int(v) for v in bot_gbw_colors[idx])
                cnt = int(bot_gbw_counts[idx])
                frac = cnt / float(bot_pixels_extracted.shape[0])
                hex_color = "#{:02x}{:02x}{:02x}".format(*col)
                
                if max(col) < 60:
                    category = "BLACK (wheels)"
                elif min(col) >= 200 and np.var(col) < 30:
                    category = "WHITE (highlights)"
                elif np.var(col) < 10:
                    category = "GRAY (body)"
                else:
                    category = "GRAY (tinted)"
                
                print(f"{i:4} | {col!s:16} | {cnt:6} | {frac:6.2%} | {hex_color} | {category}")


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def find_x_on_row(frame: np.ndarray, target_hex: str = '#464646', row: int = 470) -> int:
    """Return median x index on `row` where pixel equals target_hex. Return None if no match."""
    img = np.asarray(frame)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('frame must be HxWx3 RGB uint8')
    h, w, _ = img.shape
    if row < 0 or row >= h:
        raise ValueError('row out of bounds')
    target = np.array(hex_to_rgb(target_hex), dtype=np.uint8)
    row_pixels = img[row, :, :]
    matches = np.all(row_pixels == target, axis=1)
    xs = np.nonzero(matches)[0]
    if xs.size == 0:
        return None
    return int(np.median(xs))


def draw_orenge_dot(frame: np.ndarray, x: int, y: int, offset_x: int = 10, radius: int = 3) -> np.ndarray:
    """Draw a red filled dot at (x+offset_x, y) on a copy of frame and return it. If x is None, returns a copy unchanged."""
    img = np.asarray(frame).copy()
    h, w, _ = img.shape
    if x is None:
        return img
    dot_x = int(np.clip(x + offset_x, 0, w - 1))
    dot_y = int(np.clip(y, 0, h - 1))
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    left = max(dot_x - radius, 0)
    top = max(dot_y - radius, 0)
    right = min(dot_x + radius, w - 1)
    bottom = min(dot_y + radius, h - 1)
    draw.ellipse((left, top, right, bottom), fill=(255, 165, 0))
    return np.array(pil)


def record_bot_path_on_frames(
                              frames_path: str = 'tests/frameshots/frames',
                              frame_start: int = 0,
                              frames_jumps: int = 3,
                              page_steps: int = 23,
                              bot_step: int = 7,
                              row: int = 470,
                              target_hex: str = '#464646',
                              offset_x: int = 4,
                              radius: int = 5,
                            #   out_path: str = 'tests/frameshots/fruitbot_frame_marked.png',
                              path_number: int = 1) -> Image.Image:
    """Record bot path on frames and return the final annotated image."""

    starting_image_path = os.path.join(frames_path, f'fruitbot_frame_{frame_start*frames_jumps}.png')
    if not os.path.exists(starting_image_path):
        print("There is no image in path=", starting_image_path)
        return None
    
    print(f'Starting image path: {starting_image_path}')
    base_image = Image.open(starting_image_path).convert('RGB').resize((512, 512))
    base_frame = np.array(base_image)
    for i in range(page_steps):
        image_path = os.path.join(frames_path, f'fruitbot_frame_{frames_jumps*(i+frame_start)}.png')
        # print(f'Processing frame: {image_path}')
        if os.path.exists(image_path):
            pil_img = Image.open(image_path).convert('RGB').resize((512, 512))
        else:
            print(f'Image not found: {image_path}, skipping.')
            break
        frame = np.array(pil_img)
        cx = find_x_on_row(frame, target_hex, row=row)
        base_frame = draw_orenge_dot(base_frame, cx, row - (bot_step*frames_jumps*i), offset_x=offset_x, radius=radius)
    return Image.fromarray(base_frame)


def combine_paths(first_image_path: str, sec_image_path: str, save_path: str) -> Image.Image:
    img1 = Image.open(first_image_path).convert('RGB')
    img2 = Image.open(sec_image_path).convert('RGB')

    w1, h1 = img1.size
    w2, h2 = img2.size
    out_w = max(w1, w2)
    out_h = h1 + h2 - 25  # subtract cut_px from total height

    combined = Image.new('RGB', (out_w, out_h), (0, 0, 0))

    cut_px = 25  # pixels to remove from bottom of second image
    w2, h2 = img2.size
    img2_cropped = img2.crop((0, 0, w2, max(0, h2 - cut_px)))

    # center horizontally when pasting
    combined.paste(img2_cropped, ((out_w - w2) // 2, 0))      # sec_image on top
    combined.paste(img1, ((out_w - w1) // 2, img2_cropped.height))     # first image below
    combined.save(save_path)
    print(f"Saved combined image to {save_path}")
    return combined


def draw_full_path(frames_path: str, out_path: str):
    frames_num = len([f for f in os.listdir(frames_path) if f.startswith('fruitbot_frame_') and f.endswith('.png')])
    print(f"found {frames_num} files in folder")
    first_frame = 0
    path_num = 1
    while first_frame < frames_num:
        image = record_bot_path_on_frames(
            frames_path=frames_path,
            frame_start=first_frame,
            frames_jumps=3,
            page_steps=23,
            bot_step=7,
            row=470,
            target_hex='#464646',
            offset_x=4,
            radius=5
        )
        # display(image)
        if image is None:
            break
        image.save(out_path+f'/fruitbot_path_{path_num}.png')
        path_num += 1
        first_frame += 23
    print(f"finish with {path_num-1} paths")

    if path_num<=2:
        combine_image = Image.open(out_path+f'/fruitbot_path_1.png')
        combine_image.save(out_path+'/fruitbot_full_path_combined.png')
        return

    combine_paths(
        first_image_path=out_path+'/fruitbot_path_1.png',
        sec_image_path=out_path+'/fruitbot_path_2.png',
        save_path=out_path+'/fruitbot_full_path_combined.png'
    )

    next_path = 3
    while(next_path)<path_num:
        combine_paths(
            first_image_path=out_path+'/fruitbot_full_path_combined.png',
            sec_image_path=out_path+f'/fruitbot_path_{next_path}.png',
            save_path=out_path+'/fruitbot_full_path_combined.png'
        )
        next_path +=1


def compare_models(env1, env2, model1_path: str, model2_path: str, save_path: str):
    model1 = PPO.load(model1_path)
    model2 = PPO.load(model2_path)

    frames_path1 = save_path + '/model1_frames'
    frames_path2 = save_path + '/model2_frames'
    os.makedirs(frames_path1, exist_ok=True)
    os.makedirs(frames_path2, exist_ok=True)

    # Record frames for model 1
    def record_frames(env, model, frames_path):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = action.item() if hasattr(action, 'item') else int(action)
            obs, reward, done, info = env.step(action)
            frame = env.render()
            img = Image.fromarray(frame).convert('RGB').resize((512, 512))
            img.save(os.path.join(frames_path, f'fruitbot_frame_{step}.png'))
            step += 1
    
    record_frames(env1, model1, frames_path1)
    record_frames(env2, model2, frames_path2)


    # Draw full paths
    out_path1 = save_path + '/model1_path'
    out_path2 = save_path + '/model2_path'
    os.makedirs(out_path1, exist_ok=True)
    os.makedirs(out_path2, exist_ok=True)

    draw_full_path(frames_path1, out_path1)
    draw_full_path(frames_path2, out_path2)