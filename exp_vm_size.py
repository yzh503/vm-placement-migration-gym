import json
from src.record import Record
import main 
import yaml
import numpy as np
from multiprocessing import Pool
import pandas as pd
from os.path import exists
import copy 
from exp_config import cores, multiruns, episodes

def evaluate_seeds(agent, weightspath, seq):

    configfile = open('config/reward1.yml')
    config = yaml.safe_load(configfile)

    config['environment']['eval_steps'] = episodes
    config['environment']['sequence'] = seq

    if seq == 'lowuniform':
        config['environment']['arrival_rate'] = config['environment']['p_num']/0.375/config['environment']['service_rate']
    elif seq == 'highuniform':
        config['environment']['arrival_rate'] = config['environment']['p_num']/0.625/config['environment']['service_rate']

    args = []
    records = []
    for seed in np.arange(0, multiruns): 
        recordname = f"data/exp_vm_size/{agent}-{seq}-{seed}.json"
        if exists(recordname):
            print(recordname + ' exists')
            f = open(recordname, 'r')
            jsonstr = json.load(f)
            record = Record.import_record('ppomd', jsonstr)
            records.append(record)
            f.close()
            del jsonstr
        else: 
            print(recordname + ' does not exist.')
            config = copy.deepcopy(config)
            config['environment']['seed'] = seed
            args.append(main.Args(
                    agent=agent, 
                    config=config, 
                    silent=True,
                    logdir=None,
                    output=None,
                    jobname=None,
                    weightspath=weightspath,
                    eval=True,
                    debug=False))

    if len(args) > 0:
        with Pool(cores) as pool: 
            for record in pool.imap_unordered(main.run, args): 
                seed = record.env_config['seed']
                recordname = f"data/exp_vm_size/{agent}-{seq}-{seed}.json"
                record.save(recordname)
                records.append(record)

    returns, served_reqs, cpu, target_util, drop_rates, suspended, waiting_ratios = [], [], [], [], [], [], []
    total_suspended = []
    total_served = []
    for record in records:
        returns.append(record.total_rewards)
        served_reqs.append(record.served_requests)
        total_served.append(record.served_requests[-1])
        cpu.append(record.cpu)
        target_util.append(record.target_cpu_mean)
        drop_rates.append(record.drop_rate)
        suspended.append(record.suspended)
        total_suspended.append(record.suspended[-1])
        waiting_ratios.append(record.waiting_ratio) 
    
    returns = np.array(returns)
    cpu = np.array(cpu) # dim 0: multiple tests, dim 1: testing steps, dim 2: pms 
    target_util = np.array(target_util)
    served_reqs = np.mean(served_reqs, axis=0)
    drop_rates = np.mean(drop_rates, axis=0)

    pm_mean_multitests = np.mean(cpu, axis=2)
    pm_var_multitests = np.var(cpu, axis=2)
    pm_var = np.mean(pm_var_multitests, axis=0)

    to_print = '%s,' % (agent) 
    to_print += '%s,' % (seq) 
    to_print += '%.4f,' % (np.mean(returns))
    to_print += '%.4f,' % (np.mean(drop_rates))
    to_print += '%d,' % (np.mean(total_served))
    to_print += '%d,' % (np.mean(total_suspended))
    to_print += '%.4f,' % (np.mean(pm_mean_multitests))
    to_print += '%.4f,' % (np.mean(np.mean(target_util, axis=1)))
    to_print += '%.4f,' % (np.mean(pm_var))
    to_print += '%.4f\n' % (np.mean(waiting_ratios))

    del records

    return to_print

if __name__ == '__main__':
    print("Evaluating VM Size...")
    
    to_print = 'Model, Seq, Return, Drop Rate, Served VM, Suspend Actions, Util, Util Target, Util Var, Waiting Ratio\n'

    to_print += evaluate_seeds('ppomd', 'weights/ppomd-r1.pt', 'lowuniform')
    to_print += evaluate_seeds('firstfitmd', None, 'lowuniform')
    to_print += evaluate_seeds('bestfitmd', None, 'lowuniform')

    to_print += evaluate_seeds('ppomd', 'weights/ppomd-r1.pt', 'highuniform')
    to_print += evaluate_seeds('firstfitmd', None, 'highuniform')
    to_print += evaluate_seeds('bestfitmd', None, 'highuniform')
    
    file = open('data/exp_vm_size/summary.csv', 'w')
    file.write(to_print)
    file.close()