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
    configfile = open('config/100.yml')
    config = yaml.safe_load(configfile)
    config['environment']['reward_function'] = rewardfn
    config['environment']['arrival_rate'] = np.round(config['environment']['pms']/0.55/config['environment']['service_length'], 3)
    config['agents']['ppo']['migration_ratio'] = migration_ratio

    args = []

    recordname = 'data/exp_migration_ratio/%s-%s-%.3f.json' % (agent, rewardfn, migration_ratio)
    
    if exists(recordname):
        print(f"{recordname} exists")
        f = open(recordname, 'r')
        jsonstr = ujson.load(f)
        record = Record.import_record(agent, jsonstr)
        f.close()
    else: 
        print(f"{recordname} does not exist")
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

    to_print = '%s,' % (agent) 
    to_print += '%s,' % (rewardfn)
    to_print += '%.3f,' % (migration_ratio) 
    to_print += '%.3f,' % (np.mean(record.cpu))
    to_print += '%.3f' % (np.mean(record.slowdown_rates))
    del record
    return to_print + '\n'


def evaluate_wrapper(args, results):
    res = evaluate(args)
    results.append(res)

if __name__ == '__main__':
    to_print = 'Agent,Reward,Migration Ratio,CPU,Average Slowdown\n'
    args = []

    for migration_ratio in np.arange(0.0, 0.01, 0.001):
        args.append(('ppo', 'weights/ppo-wr.pt', 'wr', migration_ratio))
        args.append(('ppo', 'weights/ppo-ut.pt', 'ut', migration_ratio))
        args.append(('ppo', 'weights/ppo-kl.pt', 'kl', migration_ratio))
        args.append(('bestfit', None, 'ut', migration_ratio))




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


