# A Unified Approach to Virtual Machine Placement and Migration in the Cloud using Deep Reinforcement Learning

This repo contains the gym environment for VM placement and migration. You may trian DQN and PPO agents for automatic VM placement and migration in this environment.

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

To run all experiments (with 5+ CPU cores),

```shell
./run_experiments.sh
```

The experiment results are saved in `data`. When the experiments complete, draw plots in `plots.ipynb`.

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
    1. "uniform": Unif(0.1,1)
    2. "lowuniform": Unif(0.1,0.65)
    3. "highuniform": Unif(0.25,1)
