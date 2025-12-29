import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import io
import base64
from PIL import Image, ImageDraw
import time
from functools import wraps
from typing import Tuple
from stable_baselines3 import PPO
import os

import cv2
# from IPython.display import display  # Not needed in production web app


# Object types:const int BARRIER = 1;
# const int OUT_OF_BOUNDS_WALL = 2;
# const int PLAYER_BULLET = 3;
# const int BAD_OBJ = 4;
# const int GOOD_OBJ = 7;
# const int LOCKED_DOOR = 10;
# const int LOCK = 11;
# const int PRESENT = 12;
collision_object_types = [3, 4]

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
        model = PPO.load(model_path) #, env=env)
        print(f"Successfully loaded model from {zip_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {zip_path}: {e}")
        raise

def image_to_base64(img, resize=(512, 512), quality=70, format="JPEG"):
    """Convert numpy array image to base64 string for web display.
    
    Args:
        img: numpy array or PIL Image
        resize: tuple (width, height) or None to skip resizing
        quality: JPEG quality (1-100), lower = faster but more compression. 70 is good balance.
        format: 'JPEG' (faster) or 'PNG' (slower but lossless)
    """
    if img is None:
        raise ValueError("Image is None")        

    # Procgen renders as RGB array (H, W, 3)
    if isinstance(img, np.ndarray):
        # Convert to PIL Image
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img).convert('RGB')
        
        # Resize if specified
        if resize:
            pil_img = pil_img.resize(resize)
        
        # Convert to base64 (JPEG is 5-10x faster than PNG)
        buffered = io.BytesIO()
        if format == "JPEG":
            pil_img.save(buffered, format="JPEG", quality=quality, optimize=False)
        else:
            pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    elif isinstance(img, Image.Image):
        # Handle PIL Image input
        if resize:
            img = img.resize(resize)
        buffered = io.BytesIO()
        if format == "JPEG":
            img.save(buffered, format="JPEG", quality=quality, optimize=False)
        else:
            img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    else:
        raise TypeError(f"Expected numpy array or PIL Image, got {type(img)}")

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
    # display(Image.fromarray(bottom_region))  # Disabled for production

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
        # display(Image.fromarray(palette))  # Disabled for production
        
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
        # display(Image.fromarray(vis))  # Disabled for production
        
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
    return int(np.min(xs))


def draw_orenge_dot(frame: np.ndarray, x: int, y: int, offset_x: int = 10, radius: int = 5, out_color: str = 'blue', final_step: bool = False, use_rectangle: bool = False) -> np.ndarray:
    """Draw a red filled dot at (x+offset_x, y) on a copy of frame and return it. If x is None, returns a copy unchanged.
    
    Args:
        frame: Input image frame
        x: X coordinate
        y: Y coordinate
        offset_x: X offset to apply
        radius: Radius for circle or half-width/height for rectangle
        out_color: Color of outer shape ('blue', 'red', 'purple')
        final_step: Whether this is the final step
        use_rectangle: If True, draw rectangle instead of circle
    """
    img = np.asarray(frame).copy()
    h, w, _ = img.shape
    if x is None:
        return img
    offset_x = 18
    dot_x = int(np.clip(x + offset_x, 0, w - 1))
    dot_y = int(np.clip(y, 0, h - 1)) + 3

    radius_bonus = 0
    if out_color == 'red':
        radius_bonus = 10
        dot_y -= 5
    if out_color == 'purple':
        radius_bonus = 3
    # Convert to RGBA for transparency support
    pil = Image.fromarray(img).convert("RGBA")
    
    # Draw the larger semi-transparent shape first (background)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    big_radius = 24 + radius_bonus
    left_b = max(dot_x - big_radius, 0)
    top_b = max(dot_y - big_radius, 0)
    right_b = min(dot_x + big_radius, pil.size[0] - 1)
    bottom_b = min(dot_y + big_radius, pil.size[1] - 1)        
    if out_color == 'red':
        out_color = (255, 0, 0, 200)  # semi-transparent red
    elif out_color == 'purple':
        out_color = (96, 0, 128, 200)  # semi-transparent purple
    else:
        out_color = (0, 0, 255, 60)  # semi-transparent blue
    
    if use_rectangle:
        # Draw rectangle instead of circle
        ov_draw.rectangle((left_b, top_b, right_b, bottom_b), fill=out_color)
    else:
        # Draw circle
        ov_draw.ellipse((left_b, top_b, right_b, bottom_b), fill=out_color)
    
    # Composite the shape onto the image
    pil = Image.alpha_composite(pil, overlay)
    
    # Draw the smaller orange circle on top
    draw = ImageDraw.Draw(pil)
    left = max(dot_x - radius, 0)
    top = max(dot_y - radius, 0)
    right = min(dot_x + radius, w - 1)
    bottom = min(dot_y + radius, h - 1)
    draw.ellipse((left, top, right, bottom), fill=(255, 165, 0, 255))  # fully opaque orange
    
    # Convert back to RGB
    pil = pil.convert("RGB")
    
    return np.array(pil)

def draw_collision_on_image(image: np.ndarray, collision_x: float, collision_y: float, wall_collision_index: int=-1) -> np.ndarray:
    """Draw a yellow X at the object that the agent collided with on the image and return the annotated image.
    
    Args:
        image: Image array to draw on
        collision_x: Normalized X coordinate (0-1, where 0=left, 1=right)
        collision_y: Normalized Y coordinate (0-1, where 0=bottom, 1=top)
    
    Returns:
        Annotated image array with yellow X marker
    """
    if image is None:
        print("Image is None, cannot draw collision.")
        return None

    # y_offset = 160
    # if wall_collision_index >= 0:
    #     y_offset = 0
    
    # Get image dimensions
    total_height, img_width = image.shape[:2]
    print(f"Image original dimensions: width={img_width}, height={total_height}")
    img_height = 1060 # for easy world coord conversion,   image unit=53
    

    # collision_x: 0 = left edge, 1 = right edge
    # collision_y: 0 = bottom edge, 1 = top edge (need to invert for screen coords)
    x_pix = int(collision_x * img_width)
    y_pix = total_height - int((collision_y) * img_height) # y_offset + int((1.0 - collision_y) * img_height)  # Invert Y for screen coords

    print(f"Collision at normalized coords: ({collision_x:.3f}, {collision_y:.3f}) -> pixel coords: ({x_pix}, {y_pix})")
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Draw a yellow X marker at collision point
    marker_size = 10
    marker_color = (255, 255, 0)  # Yellow in RGB
    
    # Draw X shape (two diagonal lines)
    cv2.line(img_array, 
             (x_pix - marker_size, y_pix - marker_size), 
             (x_pix + marker_size, y_pix + marker_size), 
             marker_color, 3)
    cv2.line(img_array, 
             (x_pix + marker_size, y_pix - marker_size), 
             (x_pix - marker_size, y_pix + marker_size), 
             marker_color, 3)
    
    return img_array


def record_bot_path_on_image(
                              base_image: Image.Image = None,
                              frames_list: list = [],
                              frames_indexes: list = [],
                              collect_indexes: list = [],
                              collisions: list = [],
                              wall_collision_index: int = -1,
                              frame_start: int = 0,
                              frames_jumps: int = 3,
                              page_steps: int = 22,
                              bot_step: float = 7.2,
                              row: int = 470,
                              target_hex: str = '#464646',
                              offset_x: int = 6,
                              radius: int = 7,
                              path_number: int = 0,
                              frames_in_path: int = 60,
                              use_rectangle: bool = False,
                              ) -> Tuple[Image.Image, int]:
    """Record bot path on the full image and return the final annotated image.
    
    Args:
        use_rectangle: If True, draw rectangles instead of circles around dots
    """

    if base_image is None:
        print("Base image is None, cannot draw path.")
        return None
    
    base_frame = np.array(base_image)
    last_frame = 0

    # bot_cy = the bottom of the image minus some offset
    bot_cy = base_frame.shape[0] - 45
    for i in range(frame_start, len(frames_list)):
        frame = frames_list[i]
        # display(frame)
        frame_index = frames_indexes[i]
        # if frame_index > (path_number+1)*frames_in_path:
        #     print(f"Reached end of path for path_number={path_number} at frame_index={frame_index}.")
        #     break
        last_frame = i
        cx = find_x_on_row(frame, target_hex, row=row)
        final_step = False
        if frame_index in collect_indexes:
            out_color = 'blue' #'purple'
        elif frame_index == wall_collision_index:
            out_color = 'red'
            final_step = True
        else:
            out_color = 'blue'


        base_frame = draw_orenge_dot(base_frame, cx, bot_cy, offset_x=offset_x, out_color=out_color, final_step=final_step, use_rectangle=use_rectangle)
        # display(Image.fromarray(base_frame))
        if i+1 >= len(frames_indexes):
            jumps = 1
        else:
            jumps = frames_indexes[i+1] - frames_indexes[i]
        bot_cy -= bot_step*jumps

    for collision in collisions:
        base_frame = draw_collision_on_image(
            base_frame,
            collision_x=collision['x'],
            collision_y=collision['y'],
            wall_collision_index=wall_collision_index,)
    result_image = Image.fromarray(base_frame)
    
    return result_image, last_frame




def combine_paths(first_image: Image.Image, sec_image: Image.Image) -> Image.Image:
    """Combine two images vertically, cropping 25px from bottom of second image and 15px from top of first image."""
    w1, h1 = first_image.size
    w2, h2 = sec_image.size
    out_w = max(w1, w2)

    up_cut_px = 0  # pixels to remove from top of first image
    cut_px = 157  # pixels to remove from bottom of second image
    
    # Crop bottom of second image
    img2_cropped = sec_image.crop((0, 0, w2, max(0, h2 - cut_px)))
    
    # Crop top of first image
    img1_cropped = first_image.crop((0, up_cut_px, w1, h1))
    
    out_h = img1_cropped.height + img2_cropped.height

    combined = Image.new('RGB', (out_w, out_h), (0, 0, 0))

    # center horizontally when pasting
    combined.paste(img2_cropped, ((out_w - w2) // 2, 0))      # sec_image on top
    combined.paste(img1_cropped, ((out_w - w1) // 2, img2_cropped.height))     # first image below
    
    return combined


def draw_full_path(frames_list: list = [], frames_indexes: list = [], collect_indexes: list = [], collisions: list = [], frames_jumps: int = 3, wall_collision_index: int = -1, use_rectangle: bool = False) -> Tuple[Image.Image, Image.Image]:
    """Draw full path across all frames and return the combined image.
    
    Args:
        frames_list: List of frame images
        frames_indexes: List of frame indices
        collect_indexes: List of collection event indices
        frames_jumps: Number of frames to jump between recordings
        wall_collision_index: Index where wall collision occurred
        use_rectangle: If True, draw rectangles instead of circles
    """
    path_length = 50

    if frames_list is None:
        raise ValueError("Either frames_list or frames_path must be provided.")

    base_frames = []
    
    for i in range(len(frames_list)):
        frame_index = frames_indexes[i]
        if frame_index % path_length == 0:
            base_frames.append(frames_list[i])

    combined_image_clean = base_frames[0]
    for i in range(1, len(base_frames)):
        combined_image_clean = combine_paths(combined_image_clean, base_frames[i])
    

    image, _ = record_bot_path_on_image(
            base_image = combined_image_clean,
            frames_list=frames_list,
            frames_indexes=frames_indexes,
            collect_indexes=collect_indexes,
            collisions=collisions,
            wall_collision_index=wall_collision_index,
            frame_start=0,
            frames_jumps=frames_jumps,
            page_steps=22, #path_length//frames_jumps,
            row=470,
            target_hex='#464646',
            offset_x=12,
            radius=5,
            path_number=0,
            frames_in_path=path_length,
            use_rectangle=use_rectangle,
        )
    return image, combined_image_clean

    
def record_frames(env, model: PPO, frames_jumps: int = 5):# -> List[Image.Image], List[int], int:
    frames = []
    frames_indexes = []
    collect_indexes = []
    collisions = []
    wall_collision_index = -1
    frames_in_path = 60
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    done = False
    step_index = 0
    while True:
        save_frame = False
        action, _ = model.predict(obs, deterministic=True)
        action = action.item() if hasattr(action, 'item') else int(action)
        
        # Step environment
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        
        if 'collision_type' in info and (info['collision_type'] == 7 or info['collision_type'] == 4):
            collisions.append({
                'step': step_index,
                'x': info['collision_x'],
                'y': info['collision_y'],
                'collision_type': info['collision_type'],
            })
            save_frame = True

        if done:
            # save the last frame again
            img = frames[-1]
            frames.append(img)
            frames_indexes.append(step_index)
            if reward < 0:
                wall_collision_index = step_index
            break

        
        if abs(reward) > 0:
            if reward < 0 and done:
                wall_collision_index = step_index
            collect_indexes.append(step_index)
            save_frame = True

        if action == 0 or action == 2:  # Left or Right
            save_frame = True

        if save_frame or step_index % frames_jumps == 0 or step_index % frames_in_path == 0:
            # Get frame from info['rgb'] instead of render()
            frame = info.get('rgb', None)
            if frame is None:
                print(f"Warning: 'rgb' not in info at step {step_index}, using observation")
                frame = obs
            
            img = Image.fromarray(frame).convert('RGB').resize((512, 512))
            frames.append(img)
            frames_indexes.append(step_index)
        step_index += 1
        
    return frames, frames_indexes, collect_indexes, wall_collision_index, collisions

def compare_models(env1, env2, model1: PPO, model2: PPO, save_to_file: bool = False, save_path: str = ''):
    """Compare two models by recording their frames and generating path visualizations.
    
    Returns:
        tuple: (model1_path_image, model2_path_image) - PIL Images of the full paths
    """

    if save_to_file and save_path:
        frames_path1 = os.path.join(save_path, 'model1/frames')
        frames_path2 = os.path.join(save_path, 'model2/frames')
        os.makedirs(frames_path1, exist_ok=True)
        os.makedirs(frames_path2, exist_ok=True)
    else:
        # Use temp directories if not saving
        import tempfile
        temp_dir = tempfile.mkdtemp()
        frames_path1 = os.path.join(temp_dir, 'model1/frames')
        frames_path2 = os.path.join(temp_dir, 'model2/frames')
        os.makedirs(frames_path1, exist_ok=True)
        os.makedirs(frames_path2, exist_ok=True)

    frames_jumps = 2
    # Record frames for both models
    frames_list1, frames_indexes1, collect_indexes1, wall_collision_index1, collisions1 = record_frames(env1, model1, frames_jumps=frames_jumps)
    frames_list2, frames_indexes2, collect_indexes2, wall_collision_index2, collisions2 = record_frames(env2, model2, frames_jumps=frames_jumps)

    # Draw full paths
    if save_to_file and save_path:
        out_path1 = os.path.join(save_path, 'model1')
        out_path2 = os.path.join(save_path, 'model2')
        os.makedirs(out_path1, exist_ok=True)
        os.makedirs(out_path2, exist_ok=True)
    else:
        out_path1 = ''
        out_path2 = ''

    model1_path_image, _ = draw_full_path(frames_list=frames_list1, frames_indexes=frames_indexes1, collect_indexes=collect_indexes1, collisions=collisions1, frames_jumps=frames_jumps, wall_collision_index=wall_collision_index1, use_rectangle=False)
    model2_path_image, _ = draw_full_path(frames_list=frames_list2, frames_indexes=frames_indexes2, collect_indexes=collect_indexes2, collisions=collisions2, frames_jumps=frames_jumps, wall_collision_index=wall_collision_index2, use_rectangle=False)
    
    return model1_path_image, model2_path_image