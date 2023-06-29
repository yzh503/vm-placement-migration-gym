#!/bin/zsh

python main.py -e -c config/r2.yml -a firstfit -o results/firstfit.json
python main.py -e -c config/r2.yml -a bestfit -o results/bestfit.json 

# Train and evaluate dqn and ppo in the single-discrete action space using reward 1 
python main.py -e -c config/r1.yml -a ppo -w weights/ppo-r1.pt -o results/ppo-r1.json -l tensorboard -j ppo-r1 &
pid1=$!
python main.py -e -c config/r2.yml -a ppo -w weights/ppo-r2.pt -o results/ppo-r2.json -l tensorboard -j ppo-r2 &
pid2=$!
python main.py -e -c config/r3.yml -a ppo -o -w weights/ppo-r3.pt results/ppo-r1.json -l tensorboard -j ppo-r3 &
pid3=$!

# Train PPO in multi-discrete action space using reward 1, 2, and 3
python main.py -e -c config/r1.yml -a ppolstm -w weights/ppolstm-r1.pt -o results/ppolstm-r1.json -l tensorboard -j ppolstm-r1 & 
pid4=$!
python main.py -e -c config/r2.yml -a ppolstm -w weights/ppolstm-r2.pt -o results/ppolstm-r2.json -l tensorboard -j ppolstm-r2 &
pid5=$! 
python main.py -e -c config/r3.yml -a ppolstm -w weights/ppolstm-r3.pt -o results/ppolstm-r3.json -l tensorboard -j ppolstm-r3 &
pid5=$!

# Train and evaluate PPO in multi-discrete action space using reward 1, 2, and 3 on a 75% system load
python main.py -e -c config/r2-low.yml -a firstfit -o results/firstfit-low.json &
pid7=$!
python main.py -e -c config/r2-low.yml -a bestfit -o results/bestfit-low.json &
pid8=$!
python main.py -e -c config/r2-low.yml -a ppolstm -w weights/ppolstm-r2-low.pt -o results/ppolstm-r2-low.json -l tensorboard -j ppolstm-r2-low &
pid9=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6
wait $pid7
wait $pid8
wait $pid9

python exp_beta.py 
python exp_performance.py 
python exp_reward.py 
python exp_suspension.py 
python exp_vm_size.py 
python exp_var.py 