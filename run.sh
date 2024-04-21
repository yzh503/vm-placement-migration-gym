#!/bin/sh
python main.py -e -c config/10.yml -r wr -a ppo -w weights-10/ppo-wr.pt -l tensorboard -j ppo-wr &
python main.py -e -c config/10.yml -r ut -a ppo -w weights-10/ppo-ut.pt -l tensorboard -j ppo-ut &
python main.py -e -c config/10.yml -r kl -a ppo -w weights-10/ppo-kl.pt -l tensorboard -j ppo-kl &
python main.py -e -c config/10.yml -r wr -a drlvmp -w weights-10/drlvmp-wr.pt -l tensorboard -j drlvmp-wr & 
python main.py -e -c config/10.yml -r ut -a drlvmp -w weights-10/drlvmp-ut.pt -l tensorboard -j drlvmp-ut &
python main.py -e -c config/10.yml -r kl -a drlvmp -w weights-10/drlvmp-kl.pt -l tensorboard -j drlvmp-kl &

python main.py -e -c config/100.yml -r wr -a ppo -w weights/ppo-wr.pt -l tensorboard -j ppo-wr &
python main.py -e -c config/100.yml -r ut -a ppo -w weights/ppo-ut.pt -l tensorboard -j ppo-ut &
python main.py -e -c config/100.yml -r kl -a ppo -w weights/ppo-kl.pt -l tensorboard -j ppo-kl &
python main.py -e -c config/100.yml -r wr -a drlvmp -w weights/drlvmp-wr.pt -l tensorboard -j drlvmp-wr & 
python main.py -e -c config/100.yml -r ut -a drlvmp -w weights/drlvmp-ut.pt -l tensorboard -j drlvmp-ut &
python main.py -e -c config/100.yml -r kl -a drlvmp -w weights/drlvmp-kl.pt -l tensorboard -j drlvmp-kl &

wait

python exp_migration_ratio.py
python exp_reward.py
python exp_performance_small.py 
python exp_performance.py 
python exp_suspension.py 
python exp_vm_size.py 
python exp_beta.py 
python exp_convex.py