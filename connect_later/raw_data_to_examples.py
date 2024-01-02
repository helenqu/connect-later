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
SDSS_CHAR_BANDS = ['u', 'g', 'r', 'i', 'z']
SDSS_BANDS = [3561.79, 4718.87, 6185.19, 7499.7, 8961.49]
LSST_CHAR_BANDS = ['lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty']
LSST_BANDS = [3670.69, 4826.85, 6223.24, 7545.98, 8590.90, 9710.28]

def data_to_examples(infile, outfile, args=None):
    is_sdss = args.sdss if args else False

    lc_times_list = []
    with jsonlines.open(outfile, mode='w') as writer:
        print(f"Reading from {infile}, writing to {outfile}")
        if infile.suffix == '.csv':
            lightcurves = pd.read_csv(infile)
            if is_sdss:
                lightcurves['passband'] = [SDSS_CHAR_BANDS.index(x[2].lower()) for x in lightcurves['passband']] # output of snana is "b'g '"
        elif infile.suffix == '.h5':
            lightcurves = pd.read_hdf(infile, "observations")
            lightcurves = lightcurves.rename(columns={'time': 'mjd', 'flux_error': 'flux_err'})

        if is_sdss:
            lightcurves['passband'] = lightcurves['band'].apply(lambda x: SDSS_CHAR_BANDS.index(x))
        else:
            lightcurves['passband'] = lightcurves['band'].apply(lambda x: LSST_CHAR_BANDS.index(x))

        ids_in_file = np.unique(lightcurves['object_id'])
        # print(f"taking {len(ids_in_file) // 2} ids out of {len(ids_in_file)}")
        # ids_to_take = np.random.choice(ids_in_file, size=len(ids_in_file) // 2, replace=False)
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
            lc_wavelengths = [LSST_BANDS[int(x)] for x in lc['passband'].values] if not is_sdss else [SDSS_BANDS[int(x)] for x in lc['passband'].values]
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
    parser.add_argument('--outdir', help='space-delimited list of input files', required=True)
    parser.add_argument('--sdss', action='store_true', help='whether to use sdss data')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir()

    procs = []
    for infile in args.infiles:
        outfile = outdir / (Path(infile).stem + '.jsonl')
        proc = mp.Process(target=data_to_examples, args=(Path(infile), outfile, args))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join() # wait until procs are done
        print("procs done")
