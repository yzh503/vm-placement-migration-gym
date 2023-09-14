import ujson
from src.record import Record
import main 
import yaml
import numpy as np
import multiprocessing
from os.path import exists
import copy 
import exp
import time 

def evaluate(args):
    agent, weightspath, load, sr = args
    configfile = open('config/100.yml')
    config = yaml.safe_load(configfile)
    config['environment']['reward_function'] = "wr"
    config['environment']['service_length'] = sr
    config['environment']['sequence'] = "uniform"
    config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * load, 3)

    args = []
    if weightspath is None: 
        jobname = agent
    else: 
        jobname = weightspath.split('/')[-1].split('.')[0]
    recordname = 'data/exp_suspension/%s-sr%dload%.2f.json' % (jobname, sr, load)
    
    if exists(recordname):
        print(f"{recordname} exists")
        f = open(recordname, 'r')
        jsonstr = ujson.load(f)
        record = Record.import_record(agent, jsonstr)
        f.close()
    else: 
        print(f"{recordname} does not exist")
        config = copy.deepcopy(config)
        record = main.run(main.Args(
                    agent=agent, 
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

    to_print = '%s,' % (jobname) 
    to_print += '%.1f,' % (load) 
    to_print += '%d,' % (sr) 
    to_print += '%d,' % (record.served_requests[-1]) 
    to_print += '%d,' % (record.suspended[-1])
    to_print += '%d,' % (record.suspended[-1] + record.placed[-1])
    to_print += '%d,' % (np.mean(record.vm_lifetime))
    to_print += '%.3f,' % (np.mean(record.pending_rates))
    to_print += '%.3f,' % (np.mean(record.slowdown_rates))
    to_print += '%.3f' % (np.max(record.slowdown_rates))
    del record
    return to_print + '\n'


def evaluate_wrapper(args, results):
    res = evaluate(args)
    results.append(res)

if __name__ == '__main__':
    print("Evaluating Service Length and Load...")

    to_print = 'Agent, Load, Service Length, Total Served, Valid Suspend Actions, Valid Actions, Life, Average Pending, Average Slowdown, Max Slowdown\n'
    args = []

    load = 1.0
    for sr in np.arange(100, 4100, 200):
        args.append(('firstfit', None, load, sr))
        args.append(('bestfit', None, load, sr))
        args.append(('ppo', 'weights/ppo-ut.pt', load, sr))


    sr = 1000
    for load in np.arange(0.2, 1.1, 0.1):
        args.append(('firstfit', None, load, sr))
        args.append(('bestfit', None, load, sr))
        args.append(('ppo', 'weights/ppo-ut.pt', load, sr))

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

    file = open('data/exp_suspension/data.csv', 'w')
    file.write(to_print)
    file.close()


