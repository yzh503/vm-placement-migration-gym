# Get first fit and best fit result
python main.py -e -c config/reward1.yml -a firstfitmd -o results/firstfitmd.json &
python main.py -e -c config/reward1.yml -a bestfit -o results/bestfit.json 

# Train and evaluate dqn and ppo in the single-discrete action space using reward 1 
python main.py -e -c config/reward1.yml -a ppo -o results/ppo-r1.json -w weights/ppo-r1.pt -l tensorboard -j ppo-r1 &
python main.py -e -c config/reward1.yml -a dqn -o results/dqn-r1.json -w weights/dqn-r1.pt -l tensorboard -j dqn-r1 &

# Train PPO in multi-discrete action space using reward 1, 2, and 3
python main.py -e -c config/reward1.yml -a ppomd -o results/ppomd-r1.json -w weights/ppomd-r1.pt -l tensorboard -j ppomd-r1 & 
python main.py -e -c config/reward2.yml -a ppomd -o results/ppomd-r2.json -w weights/ppomd-r2.pt -l tensorboard -j ppomd-r2 & 
python main.py -e -c config/reward3.yml -a ppomd -o results/ppomd-r3.json -w weights/ppomd-r3.pt -l tensorboard -j ppomd-r3 

# Train and evaluate PPO in multi-discrete action space using reward 1, 2, and 3 on a 75% system load
python main.py -e -c config/reward1-lowload.yml -a firstfitmd -o results/firstfitmd-low.json &
python main.py -e -c config/reward1-lowload.yml -a bestfit -o results/bestfit-low.json &
python main.py -e -c config/reward1-lowload.yml -a ppomd -o results/ppomd-r1-low.json -w weights/ppomd-r1-low.pt -l tensorboard -j ppomd-r1-low 

# Collect experiment results
python exp_performance.py
python exp_reward.py
python exp_suspension.py
python exp_vm_size.py
python exp_var.py
