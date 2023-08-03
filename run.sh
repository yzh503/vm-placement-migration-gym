#!/bin/sh
python main.py -e -c config/1000.yml -r wr -a ppo -w weights/ppo-wr.pt -l tensorboard -j ppo-wr &
python main.py -e -c config/1000.yml -r ut -a ppo -w weights/ppo-ut.pt -l tensorboard -j ppo-ut &
python main.py -e -c config/1000.yml -r kl -a ppo -w weights/ppo-kl.pt -l tensorboard -j ppo-kl &
python main.py -e -c config/1000.yml -r wr -a caviglione -w weights/caviglione-wr.pt -l tensorboard -j caviglione-wr & 
python main.py -e -c config/1000.yml -r ut -a caviglione -w weights/caviglione-ut.pt -l tensorboard -j caviglione-ut &
python main.py -e -c config/1000.yml -r kl -a caviglione -w weights/caviglione-kl.pt -l tensorboard -j caviglione-kl &

wait

python exp_reward.py
python exp_migration_ratio.py
python exp_beta.py 
python exp_performance.py 
python exp_suspension.py 
python exp_vm_size.py 
python exp_var.py