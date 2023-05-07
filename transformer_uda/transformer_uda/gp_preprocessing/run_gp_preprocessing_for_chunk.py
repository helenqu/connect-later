from pathlib import Path
import yaml
import argparse
import subprocess
import multiprocessing as mp
from transformer_uda.preprocess_gp import run

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--start', type=int, help='metadata/lcdata files index to start processing at')
parser.add_argument('--end', type=int, help='metadata/lcdata files index to stop processing at')
parser.add_argument('--max_num_rows', type=int, help='metadata/lcdata files index to stop processing at')

args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.Loader)
    return config

config = load_config(args.config_path)

output_path_obj = Path(config["output_path"])
if not output_path_obj.exists():
    output_path_obj.mkdir(parents=True)

if args.end - args.start < 2: # only one file to process
    run(config, args.start, num_rows=args.max_num_rows)
else: # multiple files to process
    procs = []
    for i in range(args.start, args.end):
        proc = mp.Process(target=run, args=(config, i))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join() # wait until procs are done
        print("procs done")

    failed_procs = []
    for i, proc in enumerate(procs):
        if proc.exitcode != 0:
            failed_procs.append(i)

    if len(failed_procs) == 0:
        donefile_info = "CREATE HEATMAPS SUCCESS"
    else:
        donefile_info = "CREATE HEATMAPS FAILURE"

    donefile_path = config.get("donefile_path", output_path_obj / "done.txt")
    with open(donefile_path, "w+") as donefile:
        donefile.write(donefile_info)
