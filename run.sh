#!/bin/sh
python main.py -e -c config/r1.yml -a ppo -w weights/ppo-r1.pt -o results/ppo-r1.json -l tensorboard -j ppo-r1 &
pid1=$!
python main.py -e -c config/r2.yml -a ppo -w weights/ppo-r2.pt -o results/ppo-r2.json -l tensorboard -j ppo-r2 &
pid2=$!
python main.py -e -c config/r3.yml -a ppo -w weights/ppo-r3.pt -o results/ppo-r3.json -l tensorboard -j ppo-r3 &
pid3=$!

python main.py -e -c config/r1.yml -a rainbow -w weights/rainbow-r1.pt -o results/rainbow-r1.json -l tensorboard -j rainbow-r1 &
pid4=$!
python main.py -e -c config/r2.yml -a rainbow -w weights/rainbow-r2.pt -o results/rainbow-r2.json -l tensorboard -j rainbow-r2 &
pid5=$!
python main.py -e -c config/r3.yml -a rainbow -w weights/rainbow-r3.pt -o results/rainbow-r3.json -l tensorboard -j rainbow-r3 &
pid6=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4
wait $pid5
wait $pid6

python main.py -e -c config/r1.yml -a caviglione -w weights/ppolstm-r1.pt -o results/ppolstm-r1.json -l tensorboard -j ppolstm-r1 & 
pid7=$!
python main.py -e -c config/r2.yml -a caviglione -w weights/ppolstm-r2.pt -o results/ppolstm-r2.json -l tensorboard -j ppolstm-r2 &
pid8=$! 
python main.py -e -c config/r3.yml -a caviglione -w weights/ppolstm-r3.pt -o results/ppolstm-r3.json -l tensorboard -j ppolstm-r3 &
pid9=$!

python main.py -e -c config/r1.yml -a caviglione -w weights/caviglione-r1.pt -o results/caviglione-r1.json -l tensorboard -j caviglione-r1 & 
pid10=$!
python main.py -e -c config/r2.yml -a caviglione -w weights/caviglione-r2.pt -o results/caviglione-r2.json -l tensorboard -j caviglione-r2 &
pid11=$! 
python main.py -e -c config/r3.yml -a caviglione -w weights/caviglione-r3.pt -o results/caviglione-r3.json -l tensorboard -j caviglione-r3 &
pid12=$!

wait $pid7
wait $pid8
wait $pid9
wait $pid10
wait $pid11

# Train and evaluate PPO in multi-discrete action space using reward 1, 2, and 3 on a 75% system load
python main.py -e -c config/r2-low.yml -a caviglione -o results/caviglione-r2-low.json -l tensorboard -j caviglione-r2-low &
pid13=$!
python main.py -e -c config/r2-low.yml -a rainbow -o results/rainbow-r2-low.json -l tensorboard -j rainbow-r2-low  &
pid14=$!
python main.py -e -c config/r2-low.yml -a ppo -w weights/ppo-r2-low.pt -o results/ppo-r2-low.json -l tensorboard -j ppo-r2-low &
pid15=$!

wait $pid12
wait $pid13
wait $pid14
wait $pid15

python exp_beta.py 
python exp_performance.py 
python exp_reward.py 
python exp_suspension.py 
python exp_vm_size.py 
python exp_var.py 