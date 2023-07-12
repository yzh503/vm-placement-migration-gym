import ujson
from src.record import Record
import main 
import yaml
import numpy as np
from multiprocessing import Pool
from os.path import exists
import copy 
import exp

def evaluate(args):
    agent, weightspath, load, sr = args
    configfile = open('config/r3.yml')
    config = yaml.safe_load(configfile)
    config['environment']['pms'] = exp.pms
    config['environment']['vms'] = exp.vms
    config['environment']['eval_steps'] = exp.eval_steps
    config['environment']['reward_function'] = "waiting_ratio"
    config['environment']['service_length'] = sr
    config['environment']['sequence'] = "uniform"
    config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'] * load, 3)

    args = []
    recordname = 'data/exp_suspension/%s-sr%dload%.2f.json' % (agent, sr, load)
    
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

if __name__ == '__main__':
    print("Evaluating Service Rate and Load...")

    to_print = 'Agent, Load, Serv Rate, Total Served, Valid Suspend Actions, Valid Actions, Life, Average Pending, Average Slowdown, Max Slowdown\n'
    args = []

    load = exp.load
    for sr in np.arange(100, 4100, 200):
        args.append(('firstfit', None, load, sr))
        args.append(('bestfit', None, load, sr))
        args.append(('ppo', 'weights/ppo-r3.pt', load, sr))

    sr = exp.service_length
    for load in np.arange(0.2, 1.2, 0.1):
        args.append(('firstfit', None, load, sr))
        args.append(('bestfit', None, load, sr))
        args.append(('ppo', 'weights/ppo-r3.pt', load, sr))
    
    with Pool(exp.cores) as pool: 
        for res in pool.imap_unordered(evaluate, args): 
            to_print += res

    file = open('data/exp_suspension/data.csv', 'w')
    file.write(to_print)
    file.close()


