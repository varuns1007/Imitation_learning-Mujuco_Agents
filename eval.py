#This file is a part of COL778 A3
import os
import gym
import time
import torch
import numpy as np
import config as exp_config
import utils.utils as utils
import utils.pytorch_util as ptu
from   utils.logger import Logger

def setup_agent(args, configs):
    global env, agent
    
    env = gym.make(args.env_name,render_mode=None)
    env.action_space.seed()
    env.reset()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    
    from agents.mujoco_agents import ImitationAgent as Agent
    agent = Agent(ob_dim, ac_dim, args, **configs['hyperparameters'])

    agent.model.load_state_dict(torch.load(args.model_path))
    agent.model.eval()
    return

def eval_agent(args, configs):
    logger     = Logger(args.logdir)
    max_ep_len = configs.get("episode_len", None) or env.spec.max_episode_steps
    # set random seeds
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)
    
    total_envsteps = 0
    start_time = time.time()

    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    agent.to(ptu.device)
    
    print("\nCollecting video rollouts...")
    eval_video_trajs = utils.sample_n_trajectories(
        env, agent.get_action, args.n_vids_per_log, max_ep_len, render=True
    )

    logger.log_trajs_as_videos(
        eval_video_trajs,
        1,
        fps=fps,
        max_videos_to_save=args.n_vids_per_log,
        video_title="eval_rollouts",
    )

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", choices = ['Hopper-v4', 'Ant-v4'], type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--n_vids_per_log", type=int, default=4)
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    configs = exp_config.configs[args.env_name]

    data_path       = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data")
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    
    folder_name = "Model/"
    for i in configs['hyperparameters'].values():
        folder_name = f"{folder_name}_{i}"
    
    folder_name = f"{folder_name}_{configs['num_iteration']}"
    folder_name = f"{folder_name}_{configs['episode_len']}"

    logdir = (
        args.env_name
        + "_"
        + folder_name
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    setup_agent(args, configs)
    eval_agent(args, configs)


if __name__ == "__main__":
    main()
