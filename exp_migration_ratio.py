import ujson
from src.record import Record
import main 
import yaml
import numpy as np
import multiprocessing
from os.path import exists
import exp
import time 

def evaluate(args):
    agent, weightspath, rewardfn, migration_ratio = args
    configfile = open('config/1000.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    config['environment']['reward_function'] = rewardfn
    config['environment']['service_length'] = exp.service_length
    config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * exp.load, 3)
    config['agents']['ppo']['migration_ratio'] = migration_ratio

    if '-att-' in weightspath:
        config['agents']['ppo']['attention'] = True
        
    args = []

    recordname = 'data/exp_migration_ratio/%s-%.3f.json' % (agent, migration_ratio)
    
    if exists(recordname):
        print(f"{recordname} exists")
        f = open(recordname, 'r')
        jsonstr = ujson.load(f)
        record = Record.import_record(agent, jsonstr)
        f.close()
    else: 
        print(f"{recordname} does not exist")
        record = main.run(main.Args(
                    agent='ppo', 
                    reward=config['environment']['reward_function'],
                    config=config, 
                    silent=True,
                    logdir=None,
                    output=None,
                    jobname=None,
                    weightspath=weightspath,
                    eval=True,
                    debug=False))
        record.save(recordname)

    to_print = '%s,' % (agent) 
    to_print += '%.3f,' % (migration_ratio) 
    to_print += '%d,' % (record.served_requests[-1]) 
    to_print += '%.3f,' % (np.mean(record.cpu))
    to_print += '%.3f,' % (np.mean(record.pending_rates))
    to_print += '%.3f,' % (np.mean(record.slowdown_rates))
    to_print += '%.3f' % (np.max(record.slowdown_rates))
    del record
    return to_print + '\n'


def evaluate_wrapper(args, results):
    res = evaluate(args)
    results.append(res)

if __name__ == '__main__':
    to_print = 'Agent, Migration Discount, Total Served, CPU, Average Pending, Average Slowdown, Max Slowdown\n'
    args = []

    for migration_ratio in np.arange(0.0, 1.0, 0.05):
        args.append(('ppo-wr', 'weights/ppo-wr.pt', 'wr', migration_ratio))
        args.append(('ppo-ut', 'weights/ppo-ut.pt', 'ut', migration_ratio))
        args.append(('ppo-kl', 'weights/ppo-kl.pt', 'kl', migration_ratio))
        args.append(('ppo-kl', 'weights/ppo-att-kl.pt', 'kl', migration_ratio))


    manager = multiprocessing.Manager()
    results = manager.list()
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

        p = multiprocessing.Process(target=evaluate_wrapper, args=(arg, results))
        p.daemon = False  
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for res in results:
        to_print += res

    file = open('data/exp_migration_ratio/data.csv', 'w')
    file.write(to_print)
    file.close()


