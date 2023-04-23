import ujson
import yaml
import numpy as np
from multiprocessing import Pool
import pandas as pd
from os.path import exists
import copy
import main 
from src.record import Record
from exp_config import cores, multiruns, episodes

def evaluate_seeds(args, results):

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
    for seed in np.arange(0, multiruns): 
        recordname = 'data/exp_performance/p%ssr%dload%.2f/%s-%d.json' % (config['environment']['p_num'], sr, load, agent, seed)        
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
        with Pool(cores) as pool: 
            for record in pool.imap_unordered(main.run, args): 
                seed = record.env_config['seed']
                recordname = 'data/exp_performance/p%ssr%dload%.2f/%s-%d.json' % (config['environment']['p_num'], sr, load, agent, seed)     
                record.save(recordname)
                records.append(record)

    returns, served_reqs, pm_util, drop_rates, suspended, waiting_ratios, pending_rates, slowdown_rates = [], [], [], [], [], [], [], []
    total_suspended = []
    total_served = []
    for record in records:
        returns.append(record.total_rewards)
        served_reqs.append(record.served_requests)
        total_served.append(record.served_requests[-1])
        pm_util.append(record.pm_utilisation)
        drop_rates.append(record.drop_rate)
        suspended.append(record.suspended)
        total_suspended.append(record.suspended[-1])
        waiting_ratios.append(record.waiting_ratio) 
        pending_rates.append(np.mean(record.pending_rates))
        slowdown_rates.append(np.mean(record.slowdown_rates))
  
    returns = np.array(returns)
    pm_util = np.array(pm_util) # dim 0: multiple tests, dim 1: testing steps, dim 2: pms 
    served_reqs = np.mean(served_reqs, axis=0)
    drop_rates = np.mean(drop_rates, axis=0)
    pm_mean_multitests = np.mean(pm_util, axis=2)
    pm_var_multitests = np.var(pm_util, axis=2)
    pm_var = np.mean(pm_var_multitests, axis=0)
    
    results['agent'] += [agent] * episodes
    results['load'] += [load] * episodes
    results['service_rate'] += [sr] * episodes
    results['p_num'] += [p_num] * episodes
    results['step'] += np.arange(1, episodes + 1, 1, dtype=int).tolist()
    results['util'] += np.mean(pm_mean_multitests, axis=0).tolist()
    results['var'] += pm_var.tolist()
    results['served'] += served_reqs.tolist()
    results['suspended'] += np.mean(suspended, axis=0).tolist()
    results['waiting_ratio'] += np.mean(waiting_ratios, axis=0).tolist()
    results['slowdown_rates'] += [np.mean(slowdown_rates)] * episodes

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
    to_print += '%.3f,' % (np.mean(pm_var))
    to_print += '%.3f,' % (np.mean(pending_rates))
    to_print += '%.3f,' % (np.mean(waiting_ratios))
    to_print += '%.3f\n' % (np.mean(slowdown_rates))


    return to_print

if __name__ == '__main__':

    print("Evaluating Performance...")
    
    seq, r, tsteps = 'uniform', 1, episodes

    results = {'step': [], 'load': [], 'service_rate': [], 'p_num': [], 'agent': [], 'util': [], 'var': [], 'served': [], 'suspended': [], 'waiting_ratio': [], 'slowdown_rates': []}
    to_print = 'Agent, Reward, Load, Serv Rate, PM, Return, Drop Rate, Served VM, Suspend Actions, Util, Util Var, Pending Rate, Waiting Ratio, Slowdown Rate\n'
    
    load, sr, p_num = 1, 1000, 10
    to_print += evaluate_seeds(('firstfitmd', None, seq, r, load, sr, p_num, tsteps), results)
    to_print += evaluate_seeds(('bestfitmd', None, seq, r, load, sr, p_num, tsteps), results)
    to_print += evaluate_seeds(('ppomd', 'weights/ppomd-r1.pt', seq, r, load, sr, p_num, tsteps), results)
    to_print += evaluate_seeds(('ppo', 'weights/ppo-r1.pt', seq, r, load, sr, p_num, tsteps), results)
    to_print += evaluate_seeds(('dqn', 'weights/dqn-r1.pt', seq, r, load, sr, p_num, tsteps), results)

    load, sr, p_num = 0.75, 1000, 10
    to_print += evaluate_seeds(('firstfitmd', None, seq, r, load, sr, p_num, tsteps), results)
    to_print += evaluate_seeds(('bestfitmd', None, seq, r, load, sr, p_num, tsteps), results)
    to_print += evaluate_seeds(('ppomd', 'weights/ppomd-r1-low.pt', seq, r, load, sr, p_num, tsteps), results)

    
    
    df = pd.DataFrame(results)
    df.to_csv('data/exp_performance/data.csv')

    file = open('data/exp_performance/summary.csv', 'w')
    file.write(to_print)
    file.close()

