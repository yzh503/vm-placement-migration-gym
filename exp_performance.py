import ujson
import yaml
import numpy as np
import multiprocessing
import pandas as pd
from os.path import exists
import copy
import main 
import exp
import time
from src.record import Record

def evaluate_wrapper(args, records):
    record = main.run(args)
    records.append(record)

def evaluate(args, results):

    agent, jobname, weightspath, load = args
    configfile = open('config/wr.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    config['environment']['reward_function'] = "wr"
    config['environment']['service_length'] = exp.service_length
    config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * load, 4)

    args = []
    records = []
    for seed in np.arange(0, exp.multiruns): 
        recordname = 'data/exp_performance/load%.2f/%s-%d.json' % (load, jobname, seed)        
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
            config['environment']['seed'] = int(seed)
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

    manager = multiprocessing.Manager()
    new_records = manager.list()
    processes = []

    for arg in args:
        while len(processes) >= exp.cores:
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
                    break
            else:
                time.sleep(2) # If no process has finished yet, wait a bit and check again

        p = multiprocessing.Process(target=evaluate_wrapper, args=(arg, new_records))
        p.daemon = False  
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for record in new_records:
        seed = record.env_config['seed']
        recordname = 'data/exp_performance/load%.2f/%s-%d.json' % (load, jobname, seed)     
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
    served_reqs = np.mean(served_reqs, axis=0)
    drop_rates = np.mean(drop_rates, axis=0)
    cpu_mean_multitests = np.mean(cpu, axis=2)
    cpu_var_multitests = np.var(cpu, axis=2)
    cpu_var = np.mean(cpu_var_multitests, axis=0)
    memory_mean_multitests = np.mean(memory, axis=2)
    memory_var = np.var(memory, axis=0)
    
    results['agent'] += [jobname] * exp.eval_steps
    results['load'] += [load] * exp.eval_steps
    results['step'] += np.arange(1, exp.eval_steps + 1, 1, dtype=int).tolist()
    results['cpu_mean'] += np.mean(cpu_mean_multitests, axis=0).tolist()
    results['cpu_var'] += cpu_var.tolist()
    results['memory_mean'] += np.mean(memory_mean_multitests, axis=0).tolist()
    results['memory_var'] += memory_var.tolist()
    results['served'] += served_reqs.tolist()
    results['suspended'] += np.mean(suspended, axis=0).tolist()
    results['waiting_ratio'] += np.mean(waiting_ratios, axis=0).tolist()
    results['slowdown_rates'] += [np.mean(slowdown_rates)] * exp.eval_steps

    to_print = '%s,' % (jobname) 
    to_print += '%.2f,' % (load) 
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

    print("Evaluating Performance...")
    
    results = {'step': [], 'load': [], 'agent': [], 'cpu_mean': [], 'cpu_var': [], 'memory_mean': [], 'memory_var': [], 'served': [], 'suspended': [], 'waiting_ratio': [], 'slowdown_rates': []}
    to_print = 'Agent, Load, Return, Drop Rate, Served VM, Suspend Actions, CPU Mean, CPU Variance, Memory Mean, Memory Variance, Pending Rate, Waiting Ratio, Slowdown Rate\n'
    
    to_print += evaluate(('caviglione', 'caviglione', 'weights/caviglione-wr.pt', exp.load), results)
    to_print += evaluate(('ppo', 'ppo', 'weights/ppo-wr.pt', exp.load), results)
    to_print += evaluate(('firstfit', 'firstfit',None, exp.load), results)
    to_print += evaluate(('bestfit', 'bestfit',None, exp.load), results)
    to_print += evaluate(('convex', 'convex', None, exp.load), results)


    to_print += evaluate(('caviglione', 'caviglione', 'weights/caviglione-wr-low.pt', 0.75), results)
    to_print += evaluate(('ppo', 'ppo', 'weights/ppo-wr-low.pt', 0.75), results)
    to_print += evaluate(('firstfit', 'firstfit', None, 0.75), results)
    to_print += evaluate(('bestfit', 'bestfit', None, 0.75), results)
    to_print += evaluate(('convex', 'convex', None, 0.75), results)
    
    df = pd.DataFrame(results)
    df.to_csv('data/exp_performance/data.csv')

    file = open('data/exp_performance/summary.csv', 'w')
    file.write(to_print)
    file.close()

