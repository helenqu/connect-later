import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import pdb

# Read in the plasticc data
obs = pd.read_csv('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_train_lightcurves.csv.gz')
meta = pd.read_csv('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_train_metadata.csv.gz')
ids = obs.object_id.unique()

attempts = 100

augmented = pd.DataFrame()
for id_ in tqdm(ids):
    for _ in range(attempts):
        object_obs = obs[obs.object_id == id_]

        # I estimate the distribution of number of observations in the
        # SDSS data with a mixture of 3 gaussian distributions.
        gauss_choice = np.random.choice(3, p=[0.3, 0.65, 0.05])
        if gauss_choice == 0:
            mu = 103
            sigma = 6
        elif gauss_choice == 1:
            mu = 100
            sigma = 15
        elif gauss_choice == 2:
            mu = 200
            sigma = 25
        target_observation_count = int(
            np.clip(np.random.normal(mu, sigma), 15, 245)
        )

        # remove y band
        object_obs = object_obs[object_obs.passband != 5]

        # select active season
        lower, upper = 50, 95
        mu, sigma = 88, 10
        duration_model = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        duration = int(duration_model.rvs(1))
        midpoint = duration // 2

        num_days_on_left = int(np.random.uniform(midpoint - 10, midpoint + 10))
        num_days_on_right = duration - num_days_on_left
        peakmjd = meta[meta.object_id == id_].true_peakmjd.values[0]
        object_obs = object_obs[
            (object_obs.mjd > peakmjd + num_days_on_left) &
            (object_obs.mjd < peakmjd + num_days_on_right)
        ]

        # Get the number of observations
        n_obs = len(object_obs)
        if n_obs == 0:
            continue

        # drop at least 5% of the observations if chosen observation count is too large
        num_to_drop = n_obs - target_observation_count if n_obs > target_observation_count else n_obs * 0.05
        indices_to_drop = np.random.choice(
            object_obs.index, size=int(num_to_drop), replace=False
        )
        object_obs = object_obs.drop(indices_to_drop)

        # add SDSS-like noise
        band_noises = [
            (3.7, 0.6, 20),
            (2.5, 1, 10),
            (2.7, 0.6, 15),
            (2.9, 0.6, 27),
            (4.4, 0.5, 60)
        ]

        for band, (mean, std, offset) in enumerate(band_noises):
            band_obs = object_obs[object_obs.passband == band]

            noise = np.clip(np.random.lognormal(mean, std) + offset, 9, None)
            flux_jitter = np.random.normal(0, noise)
            pdb.set_trace()

            object_obs.loc[object_obs.passband == band, 'flux'] += flux_jitter
            object_obs.loc[object_obs.passband == band, 'flux_err'] = np.sqrt(
                object_obs.loc[object_obs.passband == band, 'flux_err']**2 + noise**2
            )

        # # make MJD values closer to SDSS - no need, mjds are normalized anyway
        # gauss_choice = np.random.choice(3, p=[0.456, 0.294, 0.25])
        # if gauss_choice == 0:
        #     start_mjd = np.random.uniform(53616, 53624)
        # elif gauss_choice == 1:
        #     start_mjd = np.random.uniform(53975, 53995)
        # elif gauss_choice == 2:
        #     start_mjd = np.random.uniform(54346, 54358)
        # diff = object_obs.mjd.min() - start_mjd
        # if diff < 0:
        #     print(f"WARNING: start mjd is larger than plasticc start mjd for object {id_}")
        # object_obs.mjd -= diff

        augmented = pd.concat([augmented, object_obs])

augmented.to_csv('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_train_lightcurves_sdss_augmented_single_fluxjitter.csv', index=False)
