# A Unified Approach to Virtual Machine Placement and Migration in the Cloud using Deep Reinforcement Learning

This repo contains the gym environment for VM placement and migration. You may trian or evaluate DQN and PPO agents for automatic VM placement and migration in this environment.



## Experiment data and plots

Experiment data are located in `data`. Plots are located in `plots`.

## Re-run Experiments

### Requirements

Python 3.10.*

```shell
pip install -r requirements.txt
```

### Examples

To see help,

```shell
python main.py -h
```

By default, the config file is `config/reward1.yml`.

To test the random agent and save results in `results/random.json`:

```shell
python main.py -a random -e -o results/random.json
```

To test the dqn agent and save results in `results/dqn.json`:

```shell
python main.py -a dqn -e -o results/dqn.json
```

Inspect and update the experiment parallellism in `exp_config.py` depending on your machine.

To run all experiments,

```shell
chmod +x run.sh
./run.sh
```

If you have less than 8 cores, modify the `run.sh` and `exp_*.py` files so that it accomodates your CPU.

The experiment data are saved in `data`. When the experiments complete, draw plots in `plots.ipynb`.

## Agents

- ppo
- ppomd
- dqn
- firstfit
- firstfitmd
- bestfit
- bestfitmd
- random

where "md" means multi-discrete action space.

## Experiments

- `exp_performance`: performance evaluation of the proposed approach against the baselines. 
- `exp_reward`: evaluation of the reward functions.
- `exp_var`: evaluation of target variance.
- `exp_suspension`: evaluation of service length and system load.
- `exp_training`: the episodic returns.
- `exp_vm_size`: evaluation of VM size. 

## Configuration

environment:

- p_num: number of PMs
- v_num: number of VMs
- var: target variance
- service_rate: service length mean for VMs
- arrival_rate: 100% system load = p_num / distribution expectation / service rate
- training_steps: number of steps in an episode in training
- eval_steps: number of steps in an episode during evaluation
- seed: integer to ensure reproducibility
- reward_function:
    1. reward 1 in the paper
    2. reward 2 in the paper
    3. reward 3 in the paper
- cap_target_util: cap the upper limit of target utilisation to 100%
- sequence:
    1. "uniform": VM size follows Unif(0.1,1)
    2. "lowuniform": VM size follows Unif(0.1,0.65)
    3. "highuniform": VM size follows Unif(0.25,1)
