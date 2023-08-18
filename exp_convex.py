import main 
import yaml
import numpy as np
from multiprocessing import Pool
import os
import copy
import exp

def evaluate(params):
    configfile = open('config/100.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    
    args = []

    for W, migration_penalty, hard_solution in params: 
        config = copy.deepcopy(config)
        config['agents']['convex']['W']  = W
        config['agents']['convex']['migration_penalty'] = migration_penalty
        config['agents']['convex']['hard_solution'] = hard_solution
        config['environment']['service_length'] = exp.service_length
        config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * exp.load, 3)

        recordname = f'data/exp_var/{W}-{migration_penalty}.json'

        args.append(main.Args(
                agent='convex', 
                reward=config['environment']['reward_function'],
                config=config, 
                silent=True,
                logdir=None,
                output=recordname,
                jobname=None,
                weightspath=None,
                eval=True,
                debug=False))

    with Pool(exp.cores) as pool: 
        summary = 'W, migration_penalty, hard_solution, migrations, VMs served\n'
        for record in pool.imap_unordered(main.run, args):
            W = record.agent_config['W'] 
            migration_penalty = record.agent_config['migration_penalty']  
            hard_solution = record.agent_config['hard_solution']
            summary += f"{W}, {migration_penalty}, {hard_solution}, {record.suspended[-1]}, {record.served_requests[-1]}\n" 
            print(f"{record.agent_config} done.")
    
    if not os.path.exists('data/exp_convex/'):
        os.makedirs('data/exp_convex/')

    with open('data/exp_convex/summary.csv', 'w') as f:
        f.write(summary)
            
if __name__ == '__main__': 
    print("Evaluating Convex Optimisation Parameters...")
    params = [(2, 1, False), (5, 1, False), (30, 1, False), (30, 5, False), (30, 10, False), (30, 20, False), (30, 30, False), (30, 30, True)]
    evaluate(params)
