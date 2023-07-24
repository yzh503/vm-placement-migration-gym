import main 
import yaml
import numpy as np
from multiprocessing import Pool
from os.path import exists
import copy
import exp

def evaluate_beta(betas, evalmode):
    configfile = open('config/r1.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    if evalmode: 
        config['environment']['service_length'] = exp.service_length
        config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * exp.load, 3)
    
    args = []

    for beta in betas: 
        config = copy.deepcopy(config)
        config['environment']['beta'] = beta

        # Only if the service length is long enough would migration be worthwhile. 
        recordname = f'data/exp_beta/{beta}.json'
        weightsname = f'data/exp_beta/{beta}.pt'

        if evalmode and exists(recordname):
            continue
            
        if not evalmode and exists(weightsname):
            continue

        args.append(main.Args(
                agent='ppo', 
                config=config, 
                silent=True,
                logdir=None,
                output=recordname,
                jobname=None,
                weightspath=weightsname,
                eval=evalmode,
                debug=False))

    with Pool(exp.cores) as pool: 
        for record in pool.imap_unordered(main.run, args):   
            if record is None: 
                print('1 trained.')
            else: 
                print(f"{record.env_config['beta']} evaluated.")

if __name__ == '__main__': 
    print("Evaluating beta...")
    betas = np.around(np.arange(0.0, 1, 0.1), decimals=2)
    evaluate_beta(betas, False)
    evaluate_beta(betas, True)
