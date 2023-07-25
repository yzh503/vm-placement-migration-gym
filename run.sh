#!/bin/sh
python main.py -e -c config/r1.yml -a ppo -w weights/ppo-r1.pt -o results/ppo-r1.json -l tensorboard -j ppo-r1 &
pid1=$!
python main.py -e -c config/r2.yml -a ppo -w weights/ppo-r2.pt -o results/ppo-r2.json -l tensorboard -j ppo-r2 &
pid2=$!
python main.py -e -c config/r3.yml -a ppo -w weights/ppo-r3.pt -o results/ppo-r3.json -l tensorboard -j ppo-r3 &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

python exp_migration_ratio.py

python main.py -e -c config/r1.yml -a caviglione -w weights/caviglione-r1.pt -o results/caviglione-r1.json -l tensorboard -j caviglione-r1 & 
pid1=$!
python main.py -e -c config/r2.yml -a caviglione -w weights/caviglione-r2.pt -o results/caviglione-r2.json -l tensorboard -j caviglione-r2 &
pid2=$! 
python main.py -e -c config/r3.yml -a caviglione -w weights/caviglione-r3.pt -o results/caviglione-r3.json -l tensorboard -j caviglione-r3 &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

python exp_beta.py 
python exp_performance.py 
python exp_reward.py 
python exp_suspension.py 
python exp_vm_size.py 
python exp_var.py 