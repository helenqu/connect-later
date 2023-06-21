import pandas as pd
import numpy as np
from astropy.table import Table
import jsonlines
import multiprocessing as mp
import argparse
from pathlib import Path
from tqdm import tqdm
import pdb

LC_LENGTH = 300
LSST_BANDS = [3670.69, 4826.85, 6223.24, 7545.98, 8590.90, 9710.28]

def data_to_examples(infile, metadata_file, outfile):
    lc_times_list = []
    with jsonlines.open(outfile, mode='w') as writer:
        print(f"Reading from {infile}, writing to {outfile}")
        # lightcurves = Table.from_pandas(pd.read_csv(infile))
        # lightcurves.add_index('object_id')
        lightcurves = pd.read_csv(infile)
        metadata = pd.read_csv(metadata_file)
        ids_in_file = np.unique(lightcurves['object_id'])
        for objid in tqdm(ids_in_file):
            lc = lightcurves[lightcurves['object_id'] == objid].sort_values('mjd').reset_index(drop=True)
            # lc.sort('mjd')
            if len(lc) == 0:
                continue
            if len(lc) > LC_LENGTH:
                start = np.random.randint(0, len(lc) - LC_LENGTH)
                lc = lc.iloc[start:start+LC_LENGTH]

            if len(lc) < LC_LENGTH:
                # lc_tensor = np.pad(lc_tensor, ((0, LC_LENGTH - len(lc)), (0, 0)), 'constant')
                # lc_times = np.pad(lc_times, (0, LC_LENGTH - len(lc)), 'constant')
                # lc_wavelengths = np.pad(lc_wavelengths, (0, LC_LENGTH - len(lc)), 'constant')
                padding_df = pd.DataFrame(np.zeros((LC_LENGTH - len(lc), len(lc.columns))), columns=lc.columns)
                # lc.update(Table(padding_df))
                lc = pd.concat([lc, padding_df], axis=0).reset_index(drop=True)

            # lc_tensor =  np.lib.recfunctions.structured_to_unstructured(lc[['flux', 'flux_err']].as_array())
            # lc_times = lc['mjd'].data
            # lc_wavelengths = lc['passband'].data
            lc_tensor =  lc[['flux', 'flux_err']].to_numpy()
            lc_times = lc['mjd'].values
            lc_wavelengths = [LSST_BANDS[int(x)] for x in lc['passband'].values]
            times_wv_tensor = np.vstack((lc_times, lc_wavelengths)).T
            # np.zeros((len(np.unique(lc_wavelengths)), len(lc_times)))

            # for i, wv in enumerate(np.unique(lc_wavelengths)):
            #     times_wv_tensor[i, np.where(lc_wavelengths == wv)] = lc_times[lc_wavelengths == wv]

            writer.write({
                'object_id': str(objid),
                'times_wv': times_wv_tensor.tolist(),
                'lightcurve': lc_tensor.tolist()
            })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--infiles', nargs='+', help='space-delimited list of input files', required=True)
    parser.add_argument('--metadata_file', help='metadata file for given input files', required=True)
    args = parser.parse_args()

    outdir = Path(args.infiles[0]).parent / "examples"
    if not outdir.exists():
        outdir.mkdir()

    data_to_examples(args.infiles[0], args.metadata_file, outdir / (Path(args.infiles[0]).stem + ".jsonl"))
    # procs = []
    # for infile in args.infiles:
    #     outfile = outdir / (Path(infile).stem + '.jsonl')
    #     proc = mp.Process(target=data_to_examples, args=(infile, args.metadata_file, outfile))
    #     proc.start()
    #     procs.append(proc)
    # for proc in procs:
    #     proc.join() # wait until procs are done
    #     print("procs done")
