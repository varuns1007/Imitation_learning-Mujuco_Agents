import gym

env = gym.make("Ant-v4", render_mode="human")  # or any MuJoCo env
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated = env.step(action)
    if terminated or truncated:
        obs = env.reset()
env.close()
