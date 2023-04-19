from dataclasses import dataclass
from src.agents.bestfit import BestFitAgent, BestFitConfig
from src.agents.bestfitmd import BestFitMDAgent, BestFitMDConfig
from src.agents.ppo import PPOAgent, PPOConfig
from src.agents.ppomd import PPOMDAgent
from src.agents.firstfit import FirstFitAgent, FirstFitConfig
from src.agents.firstfitmd import FirstFitMDAgent, FirstFitMDConfig
from src.envs.env import VmEnv, EnvConfig
from src.envs.mdenv import MultiDiscreteVmEnv
from src.agents.dqn import DQNAgent, DQNConfig
from src.agents.random import RandomAgent, RandomConfig
from src.record import Record
import src.utils
import random
import yaml, argparse
import torch
import os
import numpy as np 

@dataclass
class Args:
    agent: str
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
    if args.agent in config["agents"]:
        training_config = config["agents"][args.agent]
    else:
        training_config = {}

    torch.manual_seed(env_config['seed'])
    random.seed(env_config['seed'])
    np.random.seed(env_config['seed'])

    env = VmEnv(EnvConfig(**env_config))
    env_md = MultiDiscreteVmEnv(EnvConfig(**env_config))

    if args.agent == "random": 
        agent = RandomAgent(env, RandomConfig(**training_config))
    elif args.agent == "dqn":
        agent = DQNAgent(env, DQNConfig(**training_config))
    elif args.agent == "ppo":
        agent = PPOAgent(env, PPOConfig(**training_config))
    elif args.agent == "ppomd":
        agent = PPOMDAgent(env_md, PPOConfig(**training_config))
    elif args.agent == "firstfit":
        agent = FirstFitAgent(env, FirstFitConfig(**training_config))
    elif args.agent == "firstfitmd":
        agent = FirstFitMDAgent(env_md, FirstFitMDConfig(**training_config))
    elif args.agent == "bestfit":
        agent = BestFitAgent(env, BestFitConfig(**training_config))
    elif args.agent == "bestfitmd":
        agent = BestFitMDAgent(env_md, BestFitMDConfig(**training_config))
    else: 
        print(f"Agent cannot be {agent}")
    
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

    parser.add_argument("-a", "--agent", required=True, choices=['random', 'dqn', 'ppo', 'ppomd', 'bestfit', 'bestfitmd', 'firstfit', 'firstfitmd', 'noaction'], help = "Choose an agent to train or evaluate. \"md\" indicates multi-discrete action space.")
    parser.add_argument("-c", "--config", default='config/reward1.yml', help = "Configuration for environment and agent")
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
