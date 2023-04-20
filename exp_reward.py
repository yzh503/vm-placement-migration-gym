import ujson
import yaml
import numpy as np
from multiprocessing import Pool
from os.path import exists
import copy
import main 
from src.record import Record

TOTAL_STEPS = 150000

def evaluate_seeds(args):

    agent, weightspath, seq, r, load, sr, p_num, tsteps = args
    configfile = open('config/reward1.yml')
    config = yaml.safe_load(configfile)

    config['environment']['p_num'] = p_num
    config['environment']['v_num'] = p_num * 3
    config['environment']['sequence'] = seq
    config['environment']['reward_function'] = r
    config['environment']['service_rate'] = sr
    config['environment']['eval_steps'] = tsteps

    if seq == 'uniform':
        config['environment']['arrival_rate'] = config['environment']['p_num']/0.55/config['environment']['service_rate'] * load
    elif seq == 'multinomial':
        config['environment']['arrival_rate'] = config['environment']['p_num']/0.5/config['environment']['service_rate'] * load
    else:
        config['environment']['arrival_rate'] = config['environment']['p_num']/0.33/config['environment']['service_rate'] * load
    config['environment']['arrival_rate'] = np.round(config['environment']['arrival_rate'], 3)
    
    args = []
    records = []
    for seed in np.arange(0, 8): 
        recordname = f'data/exp_reward/{sr}/{r}-{seed}.json'     
        if exists(recordname):
            print(f"{recordname} exists")
            f = open(recordname, 'r')
            jsonstr = ujson.load(f)
            record = Record.import_record(agent, jsonstr)
            records.append(record)
            f.close()
            del jsonstr
        else: 
            print(f"{recordname} does not exist")
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
        with Pool(8) as pool: 
            for record in pool.imap_unordered(main.run, args): 
                seed = record.env_config['seed']
                recordname = f'data/exp_reward/{sr}/{r}-{seed}.json'     
                record.save(recordname)
                records.append(record)

    returns, served_reqs, pm_util, target_util, drop_rates, suspended, waiting_ratios, pending_rates, slowdown_rates = [], [], [], [], [], [], [], [], []
    total_suspended = []
    total_served = []
    for record in records:
        returns.append(record.total_rewards)
        served_reqs.append(record.served_requests)
        total_served.append(record.served_requests[-1])
        pm_util.append(record.pm_utilisation)
        target_util.append(record.target_util_mean)
        drop_rates.append(record.drop_rate)
        suspended.append(record.suspended)
        total_suspended.append(record.suspended[-1])
        waiting_ratios.append(record.waiting_ratio) 
        pending_rates.append(np.mean(record.pending_rates))
        slowdown_rates.append(np.mean(record.slowdown_rates))
  
    returns = np.array(returns)
    pm_util = np.array(pm_util) # dim 0: multiple tests, dim 1: testing steps, dim 2: pms 
    target_util = np.array(target_util)
    served_reqs = np.mean(served_reqs, axis=0)
    drop_rates = np.mean(drop_rates, axis=0)
    pm_mean_multitests = np.mean(pm_util, axis=2)
    pm_var_multitests = np.var(pm_util, axis=2)
    pm_var = np.mean(pm_var_multitests, axis=0)
    
    to_print = '%s,' % (agent) 
    to_print += '%s,' % (r)
    to_print += '%.2f,' % (load) 
    to_print += '%d,' % (sr) 
    to_print += '%d,' % (p_num) 
    to_print += '%.3f,' % (np.mean(returns))
    to_print += '%.3f,' % (np.mean(drop_rates))
    to_print += '%d,' % (np.mean(total_served))
    to_print += '%d,' % (np.mean(total_suspended))
    to_print += '%.3f,' % (np.mean(pm_mean_multitests))
    to_print += '%.3f,' % (np.mean(np.mean(target_util, axis=1)))
    to_print += '%.3f,' % (np.mean(pm_var))
    to_print += '%.3f,' % (np.mean(pending_rates))
    to_print += '%.3f,' % (np.mean(waiting_ratios))
    to_print += '%.3f\n' % (np.mean(slowdown_rates))

    return to_print

if __name__ == '__main__':

    print("Evaluating Reward Functions...")
    
    seq, r, tsteps = 'uniform', 1, TOTAL_STEPS
    to_print = 'Agent, Reward, Load, Serv Rate, PM, Return, Drop Rate, Served VM, Suspend Actions, Util, Util Target, Util Var, Pending Rate, Waiting Ratio, Slowdown Rate\n'
    
    load, sr, p_num = 1, 1000, 10
    to_print += evaluate_seeds(('ppomd', 'weights/ppomd-r1-2k.pt', seq, 1, load, sr, p_num, tsteps))
    to_print += evaluate_seeds(('ppomd', 'weights/ppomd-r2-2k.pt', seq, 2, load, sr, p_num, tsteps))
    to_print += evaluate_seeds(('ppomd', 'weights/ppomd-r3-2k.pt', seq, 3, load, sr, p_num, tsteps))

    file = open('data/exp_reward/summary.csv', 'w')
    file.write(to_print)
    file.close()

