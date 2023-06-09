import ujson
import yaml
import numpy as np
from multiprocessing import Pool
from os.path import exists
import copy
import main 
from src.record import Record
import exp

def evaluate_seeds(args):

    weightspath, rewardfn, migration_discount = args
    configfile = open('config/r3.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    config['environment']['reward_function'] = rewardfn
    config['environment']['service_length'] = exp.service_length
    config['environment']['sequence'] = "uniform"
    config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * exp.load, 3)
    config['agents']['ppo']['migration_discount'] = migration_discount

    args = []
    records = []
    for seed in np.arange(0, 5): 
        recordname = f'data/exp_migration_discount/{rewardfn}-{migration_discount}-{seed}.json'     
        if exists(recordname):
            print(f"{recordname} exists")
            f = open(recordname, 'r')
            jsonstr = ujson.load(f)
            record = Record.import_record("ppo", jsonstr)
            records.append(record)
            f.close()
            del jsonstr
        else: 
            print(f"{recordname} does not exist")
            config = copy.deepcopy(config)
            config['environment']['seed'] = int(seed)
            args.append(main.Args(
                    agent='ppo', 
                    config=config, 
                    silent=True,
                    logdir=None,
                    output=None,
                    jobname=None,
                    weightspath=weightspath,
                    eval=True,
                    debug=False))

    if len(args) > 0:
        with Pool(exp.cores) as pool: 
            for record in pool.imap_unordered(main.run, args): 
                seed = record.env_config['seed']
                recordname = f'data/exp_migration_discount/{rewardfn}-{migration_discount}-{seed}.json'     
                record.save(recordname)
                records.append(record)

    returns, served_reqs, cpu, memory, drop_rates, suspended, waiting_ratios, pending_rates, slowdown_rates = [], [], [], [], [], [], [], [], []
    total_suspended = []
    total_served = []
    for record in records:
        returns.append(record.total_rewards)
        served_reqs.append(record.served_requests)
        total_served.append(record.served_requests[-1])
        cpu.append(record.cpu)
        memory.append(record.memory)
        drop_rates.append(record.drop_rate)
        suspended.append(record.suspended)
        total_suspended.append(record.suspended[-1])
        waiting_ratios.append(record.waiting_ratio) 
        pending_rates.append(np.mean(record.pending_rates))
        slowdown_rates.append(np.mean(record.slowdown_rates))
  
    returns = np.array(returns)
    cpu = np.array(cpu) # dim 0: multiple tests, dim 1: testing steps, dim 2: pms 
    memory = np.array(memory)
    served_reqs = np.mean(served_reqs, axis=0)
    drop_rates = np.mean(drop_rates, axis=0)
    cpu_mean_multitests = np.mean(cpu, axis=2)
    cpu_var_multitests = np.var(cpu, axis=2)
    cpu_var = np.mean(cpu_var_multitests, axis=0)
    memory_mean_multitests = np.mean(memory, axis=2)
    memory_var_multitests = np.var(memory, axis=2)
    memory_var = np.mean(memory_var_multitests, axis=0)
    
    to_print = '%s,' % (rewardfn)
    to_print += '%.3f,' % (migration_discount) 
    to_print += '%.3f,' % (np.mean(returns))
    to_print += '%.3f,' % (np.mean(drop_rates))
    to_print += '%d,' % (np.mean(total_served))
    to_print += '%d,' % (np.mean(total_suspended))
    to_print += '%.3f,' % (np.mean(cpu_mean_multitests))
    to_print += '%.3f,' % (np.mean(cpu_var))
    to_print += '%.3f,' % (np.mean(memory_mean_multitests))
    to_print += '%.3f,' % (np.mean(memory_var))
    to_print += '%.3f,' % (np.mean(pending_rates))
    to_print += '%.3f,' % (np.mean(waiting_ratios))
    to_print += '%.3f\n' % (np.mean(slowdown_rates))

    return to_print

if __name__ == '__main__':

    print("Evaluating Reward Functions...")

    to_print = 'Reward Function, Migration Ratio, Return, Drop Rate, Served VM, Suspend Actions, CPU Mean, CPU Variance, Memory Mean, Memory Variance, Pending Rate, Waiting Ratio, Slowdown Rate\n'
    for migration_discount in np.arange(0.00, 1.11, 0.05):
        migration_discount = np.round(migration_discount, 2).astype(float)
        to_print += evaluate_seeds(('weights/ppo-r2.pt', "utilisation", migration_discount))
        to_print += evaluate_seeds(('weights/ppo-r3.pt', "waiting_ratio", migration_discount))

    file = open('data/exp_migration_discount/summary.csv', 'w')
    file.write(to_print)
    file.close()

