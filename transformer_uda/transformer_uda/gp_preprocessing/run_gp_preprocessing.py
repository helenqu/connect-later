from pathlib import Path
import yaml
import argparse
import subprocess
from astropy.table import Table
import numpy as np
import h5py
from transformer_uda.helpers import read_data_file

SHELLSCRIPT = """
{init_env}
python {path_to_script} --config_path {config_path} --start {start} --end {end} {max_num_rows}"""

ROOT_DIR = Path('/global/homes/h/helenqu/transformer_uda')
# HELPER FUNCTIONS
def write_config(config, config_path):
    with open(config_path, "w") as f:
        f.write(yaml.dump(config))

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.Loader)
    return config

def load_configs(config_path):
    config = load_config(config_path)
    gentype_config = load_config(ROOT_DIR.parent / "scone_wrapper" / "scone" /  "create_heatmaps" / "default_gentype_to_typename.yml")["gentype_to_typename"]
    return config, gentype_config

# get id list for each sntype
def get_ids_by_sn_name(metadata_paths, sn_type_id_to_name):
    sntype_to_ids = {}
    for metadata_path in metadata_paths:
        metadata = read_data_file(metadata_path)
        sntypes = np.unique(metadata['true_target'])
        for sntype in sntypes:
            # sn_name = sn_type_id_to_name[sntype]
            sntype = int(sntype)
            current_value = sntype_to_ids.get(sntype, np.array([]))
            sntype_to_ids[sntype] = np.concatenate((current_value, metadata[metadata['true_target'] == sntype]['object_id'].astype(np.int32)))
    return sntype_to_ids

def write_ids_to_use(ids_list_per_type, fraction_to_use, num_per_type, ids_path):
    chosen_ids = []
    for ids_list in ids_list_per_type:
        num_to_choose = int(num_per_type*fraction_to_use if num_per_type else len(ids_list)*fraction_to_use)
        chosen_ids = np.concatenate((chosen_ids, np.random.choice(ids_list, num_to_choose, replace=False)))
        print(f"writing {num_to_choose} ids out of {len(ids_list)} for this type")

    print(f"writing {len(chosen_ids)} ids for {len(ids_list_per_type)} types to {ids_path}")
    f = h5py.File(ids_path, "w")
    f.create_dataset("ids", data=chosen_ids, dtype=np.int32)
    f.close()

# do class balancing
def class_balance(categorical, max_per_type, ids_by_sn_name):
    Ia_string = "Ia" if "Ia" in ids_by_sn_name.keys() else "SNIa"
    abundances = {k:len(v) for k, v in ids_by_sn_name.items()}

    if categorical:
        num_to_choose = min(abundances.values())
    else:
        num_Ias = abundances[Ia_string]
        num_non_Ias = sum(abundances.values()) - num_Ias
        num_to_choose = min(num_Ias, num_non_Ias)

    print(f"after class balancing, choose {min(num_to_choose, max_per_type)} of each type")
    return min(num_to_choose, max_per_type)

# autogenerate some parts of config
def autofill_config(config):
    if "input_path" in config and 'metadata_paths' not in config: # write contents of input_path
        config['metadata_paths'] = [f.path for f in os.scandir(config["input_path"]) if "HEAD" in f.name]
        config['lcdata_paths'] = [path.replace("HEAD", "PHOT") for path in config['metadata_paths']]

    sn_type_id_to_name = config.get("sn_type_id_to_name", GENTYPE_CONFIG)
    config["sn_type_id_to_name"] = sn_type_id_to_name

    ids_by_sn_name = get_ids_by_sn_name(config["metadata_paths"], sn_type_id_to_name)
    print(f"sn abundances by type: {[[k,len(v)] for k, v in ids_by_sn_name.items()]}")
    config['types'] = list(ids_by_sn_name.keys())

    fraction_to_use = 1. / config.get("sim_fraction", 1)
    class_balanced = config.get("class_balanced", False)
    categorical = config.get("categorical", False)
    max_per_type = config.get("max_per_type", 100_000_000)

    print(f"class balancing {'not' if not class_balanced else ''} applied for {'categorical' if categorical else 'binary'} classification, check 'class_balanced' key if this is not desired")
    if fraction_to_use < 1 or class_balanced: # then write IDs file
        ids_path = f"{config['heatmaps_path']}/ids.hdf5"
        num_per_type = class_balance(categorical, max_per_type, ids_by_sn_name) if class_balanced else None
        write_ids_to_use(ids_by_sn_name.values(), fraction_to_use, num_per_type, ids_path)
        config["ids_path"] = ids_path

    return config

def format_sbatch_file(idx, max_num_rows):
    start = idx*NUM_SIMULTANEOUS_JOBS
    end = min(NUM_PATHS, (idx+1)*NUM_SIMULTANEOUS_JOBS)
    ntasks = end - start

    shellscript_dict = {
        "init_env": CONFIG["init_env"],
        "path_to_script": (ROOT_DIR / "transformer_uda/run_gp_preprocessing_for_chunk.py").resolve(),
        "config_path": Path(ARGS.config_path).resolve(),
        "start": start,
        "end": end,
        "max_num_rows": f"--max_num_rows {max_num_rows}" if max_num_rows else "",
    }

    with open(CONFIG['sbatch_header_path'], "r") as f:
      sbatch_script = f.read()
      is_template = "REPLACE" in sbatch_script

    if is_template:
        sbatch_script_dict = {
            "REPLACE_NAME": f"preprocess_{start}_{end}",
            "REPLACE_WALLTIME": "12:00:00",
            "REPLACE_LOGFILE": (Path(CONFIG["output_path"]) / "preprocess.log").resolve(),
            "REPLACE_MEM": "128G",
            "REPLACE_JOB": SHELLSCRIPT.format(**shellscript_dict)
        }
    else:
        sbatch_script_dict = {
            "APPEND": SHELLSCRIPT.format(**shellscript_dict)
        }
    sbatch_script = _update_header(sbatch_script, sbatch_script_dict)

    sbatch_file_path = SBATCH_FILE_PATH.format(**{"index": idx})
    with open(sbatch_file_path , "w+") as f:
        f.write(sbatch_script)
    print("start: {}, end: {}".format(start, end))
    print(f"launching job {idx} from {start} to {end}")

    return sbatch_file_path

def _update_header(header, header_dict):
    for key, value in header_dict.items():
        if key in header:
            header = header.replace(key, str(value))
    append_list = header_dict.get("APPEND")
    if append_list is not None:
        lines = header.split('\n')
        lines += append_list
        header = '\n'.join(lines)
    return header

# START MAIN FUNCTION
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--max_num_rows', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    ARGS = parser.parse_args()

    CONFIG, GENTYPE_CONFIG = load_configs(ARGS.config_path)
    OUTPUT_DIR = CONFIG["output_path"]

    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True)

    CONFIG = autofill_config(CONFIG)
    write_config(CONFIG, ARGS.config_path)

    SBATCH_FILE_PATH = str(Path(OUTPUT_DIR) / "create_heatmaps__{index}.sh")

    NUM_PATHS = len(CONFIG["lcdata_paths"])
    NUM_SIMULTANEOUS_JOBS = 1 # haswell has 32 physical cores
    MAX_FOR_SHARED_QUEUE = NUM_SIMULTANEOUS_JOBS / 2 # can only request up to half a node in shared queue

    print(f"num simultaneous jobs: {NUM_SIMULTANEOUS_JOBS}")
    print(f"num paths: {NUM_PATHS}")

    for j in range(int(NUM_PATHS/NUM_SIMULTANEOUS_JOBS)+1):
        sbatch_file = format_sbatch_file(j, ARGS.max_num_rows)
        subprocess.run(["sbatch", sbatch_file])

