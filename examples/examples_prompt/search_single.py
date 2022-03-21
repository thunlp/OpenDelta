
import os
import argparse
import random
import json
from examples_prompt.search_space import AllBackboneSearchSpace, AllDeltaSearchSpace, BaseSearchSpace, DatasetSearchSpace
import optuna
from functools import partial
from optuna.samplers import TPESampler
import shutil
import time



def objective_singleseed(args, unicode, search_space_sample  ):
    os.mkdir(f"{args.output_dir}/{unicode}")
    search_space_sample.update({"output_dir": f"{args.output_dir}/{unicode}"})

    
    with open(f"{args.output_dir}/{unicode}/this_configs.json", 'w') as fout:
        json.dump(search_space_sample, fout, indent=4,sort_keys=True)
    
    command = "CUDA_VISIBLE_DEVICES={} ".format(args.cuda_id)
    command += "python run.py "
    command += f"{args.output_dir}/{unicode}/this_configs.json"
    

    status_code = os.system(command)
    print("status_code",status_code)
    # if status_code != 0:
    #     with open(f"{args.output_dir}/{args.cuda_id}.log",'r') as flog:
    #         lastlines = " ".join(flog.readlines()[-100:])
    #         if "RuntimeError: CUDA out of memory." in lastlines:
    #             time.sleep(600)  # sleep ten minites and try again
    #             shutil.rmtree(f"{args.output_dir}/{unicode}/")
    #             return objective_singleseed(args, unicode, search_space_sample)
    #         else:
    #             raise RuntimeError("error in {}".format(unicode))



    with open(f"{args.output_dir}/{unicode}/results.json", 'r') as fret:
        results =json.load(fret)

    for filename in os.listdir(f"{args.output_dir}/{unicode}/"):
        if not filename.endswith("this_configs.json"):
            full_file_name = f"{args.output_dir}/{unicode}/{filename}"
            if os.path.isdir(full_file_name):
                shutil.rmtree(f"{args.output_dir}/{unicode}/{filename}")
            else:
                os.remove(full_file_name)

    return results['test']['test_average_metrics']
    


def objective(trial, args=None):
    search_space_sample = {}
    search_space_sample.update(BaseSearchSpace().get_config(trial, args))
    search_space_sample.update(AllBackboneSearchSpace[args.model_name]().get_config(trial, args))
    search_space_sample.update(DatasetSearchSpace(args.dataset).get_config(trial, args))
    search_space_sample.update(AllDeltaSearchSpace[args.delta_type]().get_config(trial, args))
    results = []
    for seed in [100]:
        search_space_sample.update({"seed": seed})
        unicode = random.randint(0, 100000000)
        while os.path.exists(f"{args.output_dir}/{unicode}"):
            unicode = unicode+1
        trial.set_user_attr("trial_dir", f"{args.output_dir}/{unicode}")
        res = objective_singleseed(args, unicode = unicode, search_space_sample=search_space_sample)
        results.append(res)
    ave_res = sum(results)/len(results)
    return -ave_res



    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_type")
    parser.add_argument("--dataset")
    parser.add_argument("--model_name")
    parser.add_argument("--cuda_id", type=int)
    parser.add_argument("--study_name")
    parser.add_argument("--num_trials", type=int)
    parser.add_argument("--optuna_seed", type=int, default="the seed to sample suggest point")
    args = parser.parse_args()

        
    setattr(args, "output_dir", f"outputs_search/{args.study_name}")

    study = optuna.load_study(study_name=args.study_name, storage=f'sqlite:///{args.study_name}.db', sampler=TPESampler(seed=args.optuna_seed))
    study.optimize(partial(objective, args=args), n_trials=args.num_trials)



