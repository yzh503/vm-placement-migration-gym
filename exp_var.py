import main 
import yaml
import numpy as np
from multiprocessing import Pool
from os.path import exists
import copy
from exp_config import cores

def evaluate_var(vars, evalmode):
    configfile = open('config/reward1.yml')
    config = yaml.safe_load(configfile)

    args = []

    for var in vars: 
        print(var)
        config = copy.deepcopy(config)
        config['environment']['var'] = var

        # Only if the service rate is long enough would migration be worthwhile. 
        if evalmode: 
            config['environment']['service_rate'] = 1000
            config['environment']['eval_steps'] = 40000
        
        recordname = f'data/exp_var/{var}.json'
        weightsname = f'data/exp_var/{var}.pt'

        if evalmode and exists(recordname):
            continue
            
        if not evalmode and exists(weightsname):
            continue

        args.append(main.Args(
                agent='ppomd', 
                config=config, 
                silent=True,
                logdir=None,
                output=recordname,
                jobname=None,
                weightspath=weightsname,
                eval=evalmode,
                debug=False))

    with Pool(cores) as pool: 
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
