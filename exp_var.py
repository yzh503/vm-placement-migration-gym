import main 
import yaml
import numpy as np
from multiprocessing import Pool
from os.path import exists
import copy
import exp

def evaluate_var(vars, evalmode):
    configfile = open('config/r2.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    
    args = []

    for var in vars: 
        config = copy.deepcopy(config)
        config['environment']['var'] = var

        # Only if the service rate is long enough would migration be worthwhile. 
        if evalmode: 
            config['environment']['service_length'] = exp.service_length
            config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * exp.load, 3)

        recordname = f'data/exp_var/{var}.json'
        weightsname = f'data/exp_var/{var}.pt'

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
                print('1 training done.')
            else: 
                print(f"{record.env_config['var']} done.")

if __name__ == '__main__': 
    print("Evaluating Variance...")
    vars = np.around(np.arange(0.01, 1, 0.1), decimals=2)
    evaluate_var(vars, False)
    evaluate_var(vars, True)
