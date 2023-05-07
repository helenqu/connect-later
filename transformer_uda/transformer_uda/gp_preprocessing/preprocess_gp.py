import numpy as np
from pathlib import Path
import pandas as pd
from astropy.table import Table
import yaml
import argparse
import h5py
import time
import multiprocessing as mp
from .helpers import fit_gp, get_extinction, read_fits, get_band_to_wave, read_data_file

def load_data(lcdata_path: str, metadata_paths: list[str]) -> tuple[pd.DataFrame, Table]:
    print(f'Processing file: {lcdata_path}')

    # if os.path.exists(self.finished_filenames_path):
    #     finished_filenames = pd.read_csv(self.finished_filenames_path)
    #     if os.path.basename(self.metadata_path) in finished_filenames:
    #         print("file has already been processed, exiting")
    #         sys.exit(0)

    if ".csv" in Path(lcdata_path).suffixes:
        lcdata = read_data_file(lcdata_path, return_type='table')
        metadata = pd.DataFrame()
        for metadata_path in metadata_paths:
            metadata = pd.concat([metadata, read_data_file(metadata_path)])
        survey = "LSST"

        # convert int bands identifiers to string
        lsst_bands = ['u', 'g', 'r', 'i', 'z', 'Y']
        lcdata['passband'] = [lsst_bands[x] for x in lcdata['passband']]
    else: # assume fits
        metadata, lcdata, survey = read_fits(lcdata_path)
        # metadata_ids = metadata[metadata.true_target.isin(types)].object_id
    metadata_ids = metadata.object_id

    lcdata.add_index('object_id')

    # survey info
    band_to_wave = get_band_to_wave(survey)
    lcdata['passband'] = [flt.strip() for flt in lcdata['passband']]

    expected_filters = list(band_to_wave.keys())
    lcdata = lcdata[np.isin(lcdata['passband'], expected_filters)]
    if len(lcdata) == 0:
        print("expected filters filtering not working")
        return
    lcdata['wavelength'] = [band_to_wave[flt] for flt in lcdata['passband']]

    return metadata, lcdata

def _calculate_mjd_range(sn_metadata, sn_data, mjd_minmax, has_peakmjd):
    pass

def get_all_gp_predictions(metadata, lcdata, ids, mjd_bins, index):
    #TODO: infer this from config file rather than making the subclasses pass it in
    # self.fit_on_full_lc = fit_on_full_lc

    # for output_path, mjd_minmax in zip(output_paths, mjd_minmaxes):

    done_by_type = {}
    removed_by_type = {}
    done_ids = []

    def _remove(sn_name):
        removed_by_type[sn_name] = 1 if sn_name not in removed_by_type else removed_by_type[sn_name] + 1

    def _done(sn_name, sn_id):
        done_ids.append(sn_id)
        done_by_type[sn_name] = 1 if sn_name not in done_by_type else done_by_type[sn_name] + 1

    timings = []
    start = time.time()

    data = pd.DataFrame()
    for i, sn_id in enumerate(ids):
        if i % 1000 == 0:
            print("processing {} of {}".format(i, len(ids)), flush=True)
            if i == 1000:
                time_to_1000 = time.time() - start
                print(f"job {index} took {time_to_1000} sec for 1000 objects; expected total time: {(len(ids)/1000)*time_to_1000} sec")

        sn_name, *sn_data = _get_sn_data(metadata, lcdata, sn_id)
        if sn_data[0] is None:
            _remove(sn_name)
            continue
        sn_metadata, sn_lcdata = sn_data

        gp = fit_gp(20, sn_lcdata)
        if gp == None:
            _remove(sn_name)
            continue

        milkyway_ebv = sn_metadata['mwebv'].iloc[0]
        # times = np.linspace(mjd_range[0], mjd_range[1], mjd_bins)
        times = np.linspace(min(sn_lcdata['mjd']), max(sn_lcdata['mjd']), mjd_bins)
        wavelengths = np.unique(sn_lcdata['wavelength']) # predict at wavelengths corresponding to the bands
        predictions = _get_predictions(gp, times, wavelengths, milkyway_ebv)
        predictions['object_id'] = np.repeat(sn_id, len(predictions))

        data = pd.concat([data, predictions])
        # if sn_name not in self.type_to_int_label:
        #     if self.categorical:
        #         print(f"{sn_name} not in SN_TYPE_ID_MAP?? stopping now")
        #         break
        #     self.type_to_int_label[sn_name] = 1 if sn_name == "SNIa" or sn_name == "Ia" else 0

        # z = sn_metadata['true_z'].iloc[0]
        # z_err = sn_metadata['true_z_err'].iloc[0]

        _done(sn_name, sn_id)
    return data, done_by_type

    # if not os.path.exists(self.finished_filenames_path):
    #     pd.DataFrame({"filenames": [os.path.basename(self.metadata_path)]}).to_csv(self.finished_filenames_path, index=False)
    # else:
    #     finished_filenames = pd.read_csv(self.finished_filenames_path)
    #     finished_filenames.append({"filenames": os.path.basename(self.metadata_path)}, ignore_index=True).to_csv(self.finished_filenames_path, index=False)


# HELPER FUNCTIONS
def _get_sn_data(metadata, lcdata, sn_id, mjd_minmax=[-30, 150]):
    #TODO: find a better thing to early return
    sn_metadata = metadata[metadata.object_id == sn_id]
    if sn_metadata.empty:
        print("sn metadata empty")
        return None, None

    sn_name = sn_metadata.true_target.iloc[0]
    # already_done = sn_id in self.done_ids
    # if already_done:
    #     return sn_name, None

    sn_lcdata = lcdata.loc['object_id', sn_id]['mjd', 'flux', 'flux_err', 'passband', 'wavelength']

    #TODO: used to cut on peakmjd +/- x days, but this doesn't make sense for variables
    # should we cut on detected_bool like allam does?

    # mjd_range = self._calculate_mjd_range(sn_metadata, sn_lcdata, mjd_minmax, self.has_peakmjd)
    # min_mjd, max_mjd = mjd_minmax
    # peakmjd = sn_metadata['true_peakmjd'].iloc[0]
    # mjd_range = [peakmjd+min_mjd, peakmjd+max_mjd]
    # if not mjd_range:
    #     print("mjd range is none")
    #     return sn_name, None

    # if not self.fit_on_full_lc:
    # mjds = sn_lcdata['mjd']
    # mask = np.logical_and(mjds >= mjd_range[0], mjds <= mjd_range[1])
    # if not mask.any(): # if all false
    #     print(sn_metadata[['object_id', 'true_target', 'true_peakmjd']])
    #     print(f"empty sn data (id {sn_id}) after mjd mask", mjd_range, np.min(mjds), np.max(mjds))
    #     return sn_name, None
    # sn_lcdata = sn_lcdata[mask]

    # sn_lcdata.add_row([min(sn_lcdata['mjd'])-100, 0, 0, sn_lcdata['passband'][0], sn_lcdata['wavelength'][0]])
    # sn_lcdata.add_row([max(sn_lcdata['mjd'])+100, 0, 0, sn_lcdata['passband'][0], sn_lcdata['wavelength'][0]])

    return sn_name, sn_metadata, sn_lcdata

def _get_predictions(gp, times, wavelengths, milkyway_ebv):
    # times = np.linspace(mjd_range[0], mjd_range[1], self.mjd_bins)

    # wavelengths = np.linspace(3000.0, 10100.0, self.wavelength_bins)

    ext = get_extinction(milkyway_ebv, wavelengths)
    predictions_df = pd.DataFrame(columns=['mjd', 'wavelength', 'flux', 'flux_err'])

    for i, wav in enumerate(wavelengths):
        ext = np.tile(ext[i], len(times))
        time_wavelength_grid = np.transpose([np.tile(times, 1), np.repeat([wav], len(times))])

        predictions, prediction_vars = gp(time_wavelength_grid, return_var=True)
        ext_corrected_predictions = np.array(predictions) + ext#.reshape(len(wavelengths), len(times)) + ext
        prediction_uncertainties = np.sqrt(prediction_vars)#.reshape(len(wavelengths), len(times))

        curr_df = pd.DataFrame({'mjd': times, 'wavelength': np.repeat(wav, len(times)), 'flux': ext_corrected_predictions, 'flux_err': prediction_uncertainties})
        predictions_df = pd.concat([predictions_df, curr_df])

    return predictions_df

def get_labels(metadata, ids):
    return metadata[metadata.object_id.isin(ids)][['object_id', 'true_target']].rename(columns={'true_target': 'label'})

def get_ids(config, lcdata_ids, index):
    has_ids = False
    if 'ids_path' in config:
        ids_file = h5py.File(config['ids_path'], "r")
        ids = ids_file["ids"][()] # turn this into a numpy array
        has_ids = len(ids) > 0
        print(f"example id {ids[0]}")
        ids_file.close()
    ids_for_current_file = np.intersect1d(lcdata_ids, ids) if has_ids else lcdata_ids
    print(f"job {index}: {'found' if has_ids else 'no'} ids", flush=True)
    return ids_for_current_file

def write_summary(output_path_obj, done_by_type, index, total):
    with open(output_path_obj / "done.log", "a+") as f:
        f.write(f"####### JOB {index} REPORT #######\n")
        # f.write("type name mapping to integer label used for classification: {}".format(self.type_to_int_label))
        f.write(str(done_by_type).replace("'", "") + "\n")
        total_rows = np.sum(done_by_type.values())
        f.write(f"done: {total_rows} out of {total} expected\n")

def _run(metadata, lcdata, ids_for_current_file, output_file_path, file_index, rows_index):
    index = f"{file_index}_{rows_index}"

    ids_for_current_file = np.intersect1d(ids_for_current_file, metadata['object_id'])
    print(f"expect {len(ids_for_current_file)}/{len(metadata)} objects for this chunk", flush=True)

    gp_interpolated_data, done_by_type = get_all_gp_predictions(metadata, lcdata, ids_for_current_file, 180, index)
    print(f"writing {len(gp_interpolated_data)} rows to {str(output_file_path)}")

    output_path_obj = Path(output_file_path)
    gp_interpolated_data.to_csv(output_path_obj / f"preprocessed_{index}.csv", index=False)

    write_summary(output_path_obj, done_by_type, index, len(ids_for_current_file))

def run(config, index, num_rows=None):
    output_path = config['output_path']
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        output_path_obj.mkdir(parents=True)

    print(f"writing to {output_path}", flush=True)
    metadata, lcdata = load_data(config['lcdata_paths'][index], config['metadata_paths'])
    lcdata_ids = np.unique(lcdata['object_id'])
    ids_for_current_file = get_ids(config, lcdata_ids, index)
    metadata = metadata[metadata.object_id.isin(ids_for_current_file)]

    if num_rows is not None: # do multiprocessing
        print(f"running multiprocessing with {num_rows} rows per proc, {len(metadata)} total rows", flush=True)
        print(f"{range(len(metadata) // num_rows)}")
        procs = []
        for rows_index in range(len(metadata) // num_rows + 1):
            end_row = min((rows_index + 1) * num_rows, len(metadata))
            print(f"index {rows_index}, end row: {(rows_index + 1) * num_rows}, {len(metadata)}")
            print(f"starting proc {rows_index} for rows {rows_index*num_rows} : {end_row}", flush=True)
            curr_metadata = metadata.iloc[rows_index*num_rows : end_row]
            proc = mp.Process(target=_run, args=(curr_metadata, lcdata, ids_for_current_file, output_path, index, rows_index))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()
            print("procs done")
    else:
        _run(metadata, lcdata, ids_for_current_file, index, 0)

    get_labels(metadata, ids_for_current_file).to_csv(output_path_obj / f"labels_{index}.csv", index=False)
