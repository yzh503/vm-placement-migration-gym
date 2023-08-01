from dataclasses import dataclass
from src.agents.bestfit import BestFitAgent
from src.agents.caviglione import CaviglioneAgent, CaviglioneConfig
from src.agents.ppo import PPOAgent, PPOConfig
from src.agents.firstfit import FirstFitAgent
from src.agents.convex import ConvexAgent, ConvexConfig
from src.vm_gym.envs.env import EnvConfig
from src.record import Record
import gymnasium as gym
import yaml, argparse
import numpy as np 
import src.vm_gym
import src.utils
import random
import torch
import os

@dataclass
class Args:
    agent: str
    reward: str
    config: dict
    logdir: str
    output: str
    silent: bool
    jobname: str
    weightspath: str
    eval: bool
    debug: bool

def run(args: Args) -> Record:
    
    config = args.config
    env_config = config["environment"]
    env_config['reward_function'] = args.reward
    if args.agent in config["agents"]:
        agent_config = config["agents"][args.agent]
    else:
        agent_config = {}

    torch.manual_seed(env_config['seed'])
    random.seed(env_config['seed'])
    np.random.seed(env_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_config['seed'])
    torch.set_float32_matmul_precision('high')

    env = gym.make("VmEnv-v1", config=EnvConfig(**env_config))

    if args.agent == "caviglione":
        agent = CaviglioneAgent(env, CaviglioneConfig(**agent_config))
    elif args.agent == "ppo":
        agent = PPOAgent(env, PPOConfig(**agent_config))
    elif args.agent == "convex":
        agent = ConvexAgent(env, ConvexConfig(**agent_config))
    elif args.agent == "firstfit":
        agent = FirstFitAgent(env)
    elif args.agent == "bestfit":
        agent = BestFitAgent(env)
    else: 
        print(f"Agent cannot be {args.agent}")
    
    if args.logdir and args.jobname:
        agent.set_log(jobname=args.jobname, logdir=args.logdir)

    if args.weightspath:
        print(f"Weights: {args.weightspath}...")
        if os.path.exists(args.weightspath):
            agent.load_model(args.weightspath)
        else: 
            src.utils.ensure_parent_dirs_exist(args.weightspath)
            agent.learn()
    else: 
        agent.learn()
    
    if args.weightspath and not os.path.exists(args.weightspath): 
        agent.save_model(args.weightspath)
        print(f"Weights saved to {args.weightspath}.")
    
    if args.eval:
        show = not args.silent
        record = agent.test(show=show, output=args.output, debug=args.debug)
    else: 
        record = None

    agent.end_log()

    return record

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--agent", required=True, choices=["ppo", "firstfit", "bestfit", "convex", "rainbow", "caviglione"], help = "Choose an agent to train or evaluate.")
    parser.add_argument("-c", "--config", default='config/kl.yml', help = "Configuration for environment and agent")
    parser.add_argument("-r", "--reward", default='wr', choices=["wr", "ut", "kl"], help = "wr: waiting ratio, ut: utilization, kl: kl divergence")
    parser.add_argument("-d", "--debug", action='store_true', help="Print step-by-step debug info")
    parser.add_argument("-l", "--logdir", help="Directory of tensorboard logs")
    parser.add_argument("-j", "--jobname", help="Job name in tensorboard")
    parser.add_argument("-o", "--output", default='./output.json', help="Path of result json file")
    parser.add_argument("-w", "--weightspath", help="path of dqn or ppo's weights to load or to save")
    parser.add_argument("-e", "--eval", action='store_true', help="to evaluate a model instead of training")
    parser.add_argument("-s", "--silent", default=False, action='store_true', help="Do not print summary of the model")

    args = parser.parse_args()

    f = open(args.config)
    args.config = yaml.safe_load(f)
    f.close()

    run(args)
