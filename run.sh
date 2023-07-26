#!/bin/sh
python main.py -e -c config/wr.yml -a ppo -w weights/ppo-wr.pt -o results/ppo-wr.json -l tensorboard -j ppo-wr &
pid1=$!
python main.py -e -c config/ut.yml -a ppo -w weights/ppo-ut.pt -o results/ppo-ut.json -l tensorboard -j ppo-ut &
pid2=$!
python main.py -e -c config/kl.yml -a ppo -w weights/ppo-kl.pt -o results/ppo-kl.json -l tensorboard -j ppo-kl &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

python exp_migration_ratio.py

python main.py -e -c config/wr.yml -a caviglione -w weights/caviglione-wr.pt -o results/caviglione-wr.json -l tensorboard -j caviglione-wr & 
pid1=$!
python main.py -e -c config/ut.yml -a caviglione -w weights/caviglione-ut.pt -o results/caviglione-ut.json -l tensorboard -j caviglione-ut &
pid2=$! 
python main.py -e -c config/kl.yml -a caviglione -w weights/caviglione-kl.pt -o results/caviglione-kl.json -l tensorboard -j caviglione-kl &
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