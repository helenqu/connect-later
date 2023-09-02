
import pandas as pd
import numpy as np
import jsonlines
import multiprocessing as mp
import argparse
from pathlib import Path
from tqdm import tqdm

def data_to_examples(infile, outfile):
    with jsonlines.open(outfile, mode='w') as writer:
        print(f"Reading from {infile}, writing to {outfile}")
        lightcurves = pd.read_csv(infile)
        for objid in tqdm(np.unique(lightcurves['object_id'])):
            lc = lightcurves[lightcurves['object_id'] == objid]

            lc_grouped = lc.groupby('wavelength')
            groups = list(lc_grouped.groups.items())
            lc_shape = (len(groups), len(groups[0][1]))
            lc_tensor = np.zeros(lc_shape)

            lc_grouped_sorted = lc_grouped.apply(lambda x: x.sort_values('mjd').reset_index(drop=True))

            for i in range(len(np.unique(lc_grouped_sorted['wavelength']))):#, group in enumerate(lc_grouped.groups.keys()):
# lc_tensor[i] = lc_grouped.get_group(group).flux.values
                lc_tensor[i] = lc_grouped_sorted.iloc[i*lc_shape[1]:(i+1)*lc_shape[1]].flux.values

            writer.write({
                'object_id': str(objid),
                'start': lc_grouped_sorted.iloc[0].mjd,
                'lightcurve': lc_tensor.tolist()
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--infiles', nargs='+', help='space-delimited list of input files', required=True)
    args = parser.parse_args()

    procs = []
    for infile in args.infiles:
        outfile = Path(infile).parent / 'examples' / (Path(infile).stem + '.jsonl')
        proc = mp.Process(target=data_to_examples, args=(infile, outfile))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join() # wait until procs are done
        print("procs done")
