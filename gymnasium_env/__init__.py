from gymnasium.envs.registration import register

register(
    id="gymnasium_env/vizdoom_project-basic",
    entry_point="gymnasium_env.envs:VizdoomGymnasiumEnv",
)
