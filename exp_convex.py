import main 
import yaml
import numpy as np
from multiprocessing import Pool
import os
import copy
import exp
from os.path import exists
import ujson
from src.record import Record

def evaluate(params):
    configfile = open('config/10.yml')
    config = yaml.safe_load(configfile)
    
    args = []
    records = []
    for W in params: 
        config = copy.deepcopy(config)
        config['agents']['convex']['W']  = W
        config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'], 3)

        recordname = f'data/exp_convex/{W}.json'
        if exists(recordname):
            print(f"{recordname} exists")
            f = open(recordname, 'r')
            jsonstr = ujson.load(f)
            record = Record.import_record('convex', jsonstr)
            records.append(record)
            f.close()
            del jsonstr
        else:
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
        summary = 'W, migrations, VMs served\n'
        for record in pool.imap_unordered(main.run, args):
            W = record.agent_config['W'] 
            summary += f"{W}, {record.suspended[-1]}, {record.served_requests[-1]}\n" 
            print(f"{record.agent_config} done.")
        for record in records: 
            W = record.agent_config['W'] 
            summary += f"{W}, {record.suspended[-1]}, {record.served_requests[-1]}\n" 
            print(f"{record.agent_config} done.")
    
    if not os.path.exists('data/exp_convex/'):
        os.makedirs('data/exp_convex/')

    with open('data/exp_convex/summary.csv', 'w') as f:
        f.write(summary)
            
if __name__ == '__main__': 
    print("Evaluating Convex Optimisation Parameters...")
    params = [10, 20, 30]
    evaluate(params)
