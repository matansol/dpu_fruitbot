from gym.envs.registration import register
from gym3 import ToGymEnv, ViewerWrapper, ExtractDictObWrapper
from .env import ENV_NAMES, ProcgenGym3Env
from .wrappers import InfoRgbRenderWrapper, ReducedActionWrapper, StayBonusWrapper
from .gym_adapter import RenderableToGymEnv


def make_env(render_mode=None, render=False, 
             use_discrete_action_wrapper=None,
             use_stay_bonus_wrapper=None,
             use_render_wrapper=True,
             stay_bonus=0.1,
             env_name=None,
             **kwargs):
    print("starting make_env with env_name:", env_name)
    if render:
        render_mode = "human"

    use_viewer_wrapper = False
    kwargs["render_mode"] = "rgb_array"

    if render_mode == "human":
        use_viewer_wrapper = True

    # Auto-enable for fruitbot
    if env_name == "fruitbot":
        if use_discrete_action_wrapper is None:
            use_discrete_action_wrapper = True
        if use_stay_bonus_wrapper is None:
            use_stay_bonus_wrapper = True
    else:
        if use_discrete_action_wrapper is None:
            use_discrete_action_wrapper = False
        if use_stay_bonus_wrapper is None:
            use_stay_bonus_wrapper = False

    # --- gym3 env + gym3 wrappers ---
    env = ProcgenGym3Env(num=1, num_threads=0, env_name=env_name, **kwargs)

    if use_stay_bonus_wrapper:
        env = StayBonusWrapper(env, stay_bonus=stay_bonus)

    env = InfoRgbRenderWrapper(env)

    if use_viewer_wrapper:
        env = ViewerWrapper(env, tps=15, info_key="rgb")
        print("ViewerWrapper applied")
    env = ExtractDictObWrapper(env, key="rgb")
    print("ExtractDictObWrapper applied")
    # --- convert to gym.Env ---
    gym_env = ToGymEnv(env)

    # --- gym-level reduced action wrapper ---
    if use_discrete_action_wrapper:
        gym_env = ReducedActionWrapper(gym_env, valid_actions=[1, 4, 7])

    return gym_env


def register_environments():
    for env_name in ENV_NAMES:
        register(
            id=f'procgen-{env_name}-v0',
            entry_point='procgen.gym_registration:make_env',
            kwargs={"env_name": env_name},
        )