"""Implementation of avocado components for the PLAsTiCC dataset"""

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.special import erf
from scipy import stats
from collections import defaultdict
import pdb

from avocado.augment import Augmentor
from avocado.utils import AvocadoException, logger

sdss_bands = ["u", "g", "r", "i", "z"]

class SDSSAugmentor(Augmentor):
    """Implementation of an Augmentor for the PLAsTiCC dataset to emulate real SDSS data"""
    def __init__(self):
        super().__init__()

        self._photoz_reference = None

        # Load the photo-z model
        self._load_photoz_reference()

    def _load_photoz_reference(self):
        """Load the full SDSS dataset as a reference for photo-z
        estimation.

        This reads the test set and extracts all of the photo-zs and true
        redshifts. The results are cached as self._photoz_reference.

        Returns
        =======
        photoz_reference numpy ndarray
            A Nx3 array with reference photo-zs for each entry with a spec-z in
            the test set. The columns are spec-z, photo-z and photo-z error.
        """
        if self._photoz_reference is None:
            logger.info("Loading photoz reference...")

            data_path = "/pscratch/sd/h/helenqu/sdss/sdss_phot.h5"
            metadata = pd.read_hdf(data_path, "metadata")

            result = np.vstack(
                [
                    metadata["redshift"], # all host speczs are missing, these are probably spec-z of the object itself
                    metadata["host_photoz"],
                    metadata["host_photoz_error"],
                ]
            ).T

            self._photoz_reference = result

        return self._photoz_reference

    def _simulate_photoz(self, redshift):
        """Simulate the photoz determination for a lightcurve using the test
        set as a reference.

        I apply the observed differences between photo-zs and spec-zs directly
        to the new redshifts. This does not capture all of the intricacies of
        photo-zs, but it does ensure that we cover all of the available
        parameter space with at least some simulations.

        Parameters
        ----------
        redshift : float
            The new true redshift of the object.

        Returns
        -------
        host_photoz : float
            The simulated photoz of the host.

        host_photoz_error : float
            The simulated photoz error of the host.
        """
        photoz_reference = self._load_photoz_reference()

        while True:
            ref_idx = np.random.choice(len(photoz_reference))
            ref_specz, ref_photoz, ref_photoz_err = photoz_reference[ref_idx]

            # Randomly choose the order for the difference. Degeneracies work
            # both ways, so even if we only see specz=0.2 -> photoz=3.0 in the
            # data, the reverse also happens, but we can't get spec-zs at z=3
            # so we don't see this.
            new_diff = (ref_photoz - ref_specz) * np.random.choice([-1, 1])

            # Apply the difference, and make sure that the photoz is > 0.
            new_photoz = redshift + new_diff
            if new_photoz < 0:
                continue

            # Add some noise to the error so that the classifier can't focus in
            # on it.
            new_photoz_err = ref_photoz_err * np.random.normal(1, 0.05)

            break

        return new_photoz, new_photoz_err

    def _augment_redshift(self, reference_object, augmented_metadata):
        """Choose a new redshift and simulate the photometric redshift for an
        augmented object

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        augmented_metadata : dict
            The augmented metadata to add the new redshift too. This will be
            updated in place.
        """
        # Choose a new redshift.
        template_redshift = reference_object.metadata["redshift"]

        # First, we limit the redshift range as a multiple of the original
        # redshift. We avoid making templates too much brighter because
        # the lower redshift templates will be left with noise that is
        # unrealistic. We also avoid going to too high of a relative
        # redshift because the templates there will be too faint to be
        # detected and the augmentor will waste a lot of time without being
        # able to actually generate a template.
        min_redshift = 0.3 * template_redshift
        max_redshift = 1.1 * template_redshift

        # Second, for high-redshift objects, we add a constraint to make
        # sure that we aren't evaluating the template at wavelengths where
        # the GP extrapolation is unreliable.
        max_redshift = np.min([max_redshift, 1.5 * (1 + template_redshift) - 1])

        # Choose new redshift from a log-uniform distribution over the
        # allowable redshift range.
        # aug_redshift = np.exp(
        #     np.random.uniform(np.log(min_redshift), np.log(max_redshift))
        # )
        aug_redshifts = 1/np.random.power(6, size=10)-1 # modeled off the SDSS photometric sample
        aug_redshifts_in_range = aug_redshifts[(aug_redshifts > min_redshift) & (aug_redshifts < max_redshift)]
        if len(aug_redshifts_in_range) > 0:
            aug_redshift = np.random.choice(aug_redshifts_in_range)
        else:
            aug_redshift = min_redshift + np.random.normal(0, 0.1)*min_redshift

        # Simulate a new photometric redshift
        aug_photoz, aug_photoz_error = self._simulate_photoz(aug_redshift)

        augmented_metadata["redshift"] = aug_redshift
        augmented_metadata["host_specz"] = aug_redshift
        augmented_metadata["host_photoz"] = aug_photoz
        augmented_metadata["host_photoz_error"] = aug_photoz_error
        augmented_metadata["augment_brightness"] = 0.0

    def _augment_metadata(self, reference_object):
        """Generate new metadata for the augmented object.

        This method needs to be implemented in survey-specific subclasses of
        this class. The new redshift, photoz, coordinates, etc. should be
        chosen in this method.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """

        # nothing is done here since we're not doing redshifting
        augmented_metadata = reference_object.metadata.copy()
        # Choose a new redshift.
        self._augment_redshift(reference_object, augmented_metadata)

        # Smear the mwebv value a bit so that it doesn't uniquely identify
        # points. I leave the position on the sky unchanged (ra, dec, etc.).
        # Don't put any of those variables directly into the classifier!
        augmented_metadata["mwebv"] *= np.random.normal(1, 0.1)

        return augmented_metadata

    def _choose_sampling_times(
        self,
        reference_object,
        augmented_metadata,
        max_time_shift=50,
        block_width=250,
        window_padding=100,
        drop_fraction=0.1,
    ):
        """Choose the times at which to sample for a new augmented object.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        augmented_metadata : dict
            The augmented metadata
        max_time_shift : float (optional)
            The new sampling times will be shifted by up to this amount
            relative to the original ones.
        block_width : float (optional)
            A block of observations with a width specified by this parameter
            will be dropped.
        window_padding : float (optional)
            Observations outside of a window bounded by the first and last
            observations in the reference objects light curve with a padding
            specified by this parameter will be dropped.
        drop_fraction : float (optional)
            This fraction of observations will always be dropped when creating
            the augmented light curve.

        Returns
        =======
        sampling_times : pandas Dataframe
            A pandas Dataframe that has the following columns:

            - time : the times of the simulated observations.
            - band : the bands of the simulated observations.
            - reference_time : the times in the reference light curve that
              correspond to the times of the simulated observations.
        """
        reference_observations = reference_object.observations
        sampling_times = reference_observations[["time", "band"]].copy()
        sampling_times["reference_time"] = sampling_times["time"].copy()

        # remove y band observations
        sampling_times = sampling_times[sampling_times["band"] != "lssty"]

        augmented_redshift = augmented_metadata["redshift"]
        reference_redshift = reference_object.metadata["redshift"]
        redshift_scale = (1 + augmented_redshift) / (1 + reference_redshift)

        if augmented_redshift != reference_redshift:
            # Shift relative to an approximation of the peak flux time so that
            # we generally keep the interesting part of the light curve in the
            # frame.
            ref_peak_time = reference_observations["time"].iloc[
                np.argmax(reference_observations["flux"].values)
            ]

            sampling_times["time"] = ref_peak_time + redshift_scale * (
                sampling_times["time"] - ref_peak_time
            )

        if len(sampling_times) == 0:
            return sampling_times

        target_mjd_count = self._choose_target_mjd_count()
        num_mjds = len(sampling_times["time"].unique())

        num_fill = int(target_mjd_count * (redshift_scale - 1))
        if num_fill > 0:
            new_times = np.random.uniform(
                sampling_times["time"].min(),
                sampling_times["time"].max(),
                num_fill,
            )

            new_times = pd.DataFrame(
                {
                    "time": new_times,
                    "reference_time": new_times,
                    "band": np.random.choice(sampling_times['band'], size=len(new_times))
                }
            )
            # add each band for every time in new_times
            new_bands = defaultdict(list)
            for obs in new_times.itertuples():
                unimaged_bands = np.setdiff1d(sdss_bands, obs.band)
                new_bands['band'].extend(unimaged_bands)
                new_bands['time'].extend([obs.time] * len(unimaged_bands))
                new_bands['reference_time'].extend([obs.reference_time] * len(unimaged_bands))
            sampling_times = pd.concat([sampling_times, pd.DataFrame(new_bands)])

        # drop down to target number of observations or drop random 10%
        num_drop = int(
            max(
                num_mjds - target_mjd_count,
                drop_fraction * num_mjds,
            )
        )
        mjds_to_drop = np.random.choice(
            sampling_times["time"].unique(), num_drop, replace=False
        )
        sampling_times = sampling_times[~sampling_times["time"].isin(mjds_to_drop)]

        return sampling_times

    def _choose_target_mjd_count(self):
        """Choose the target number of observations for a new augmented light
        curve.

        We use a functional form that roughly maps out the number of
        observations in the SDSS dataset.

        Parameters
        ----------
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        target_mjd_count : int
            The target number of observed mjds in the new light curve.
        """
        gauss_choice = np.random.choice(3, p=[0.5, 0.45, 0.05])
        if gauss_choice == 0:
            mu = 21
            sigma = 2
        elif gauss_choice == 1:
            mu = 19
            sigma = 3
        elif gauss_choice == 2:
            mu = 40
            sigma = 5
        target_mjd_count = int(np.random.normal(mu, sigma))

        return target_mjd_count

    def _simulate_light_curve_uncertainties(self, observations, augmented_metadata):
        """Simulate the observation-related noise and detections for a light
        curve.

        For the PLAsTiCC dataset, we estimate the measurement uncertainties for
        each band with a lognormal distribution for both the WFD and DDF
        surveys. Those measurement uncertainties are added to the simulated
        observations.

        Parameters
        ----------
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        # Make a copy so that we don't modify the original array.
        observations = observations.copy()

        if len(observations) == 0:
            # No data, skip
            return observations

        # add noise
        band_noises = [
            (3.3, 1, 35),
            (2.3, 1, 10),
            (2.5, 0.9, 17),
            (2.8, 0.7, 27),
            (4.1, 0.7, 75)
        ]

        for i, (mean, std, offset) in enumerate(band_noises):
            band_obs = observations[observations["band"] == sdss_bands[i]]
            # noise = np.clip(np.random.lognormal(mean, std, size=len(band_obs)) + offset, 9, None)
            # flux_jitter = np.random.normal([0]*len(band_obs), 0.1*band_obs['flux_error'].values)

            # not sure if i should make n samples of the distribution or just 1
            noise = np.random.lognormal(mean, std) + offset
            flux_jitter = np.random.normal(0, noise)

            observations.loc[observations.band == sdss_bands[i], 'flux'] += flux_jitter
            observations.loc[observations.band == sdss_bands[i], 'flux_error'] = np.sqrt(
                observations.loc[observations.band == sdss_bands[i], 'flux_error']**2 + noise**2
            )

        return observations

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.

        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(observations["flux"]) / observations["flux_error"]
        observations["detected"] = (s2n >= 5).astype(int)

        pass_detection = np.sum(observations["detected"]) >= 2 and len(np.unique(observations["time"])) > 5

        return observations, pass_detection


class PlasticcToSDSSAugmentor(Augmentor):
    """Implementation of an Augmentor for the PLAsTiCC dataset to emulate real SDSS data"""

    def __init__(self):
        super().__init__()

    def _augment_metadata(self, reference_object):
        """Generate new metadata for the augmented object.

        This method needs to be implemented in survey-specific subclasses of
        this class. The new redshift, photoz, coordinates, etc. should be
        chosen in this method.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.

        Returns
        =======
        augmented_metadata : dict
            The augmented metadata
        """

        # nothing is done here since we're not doing redshifting
        return reference_object.metadata.copy()

    def _choose_sampling_times(
        self,
        reference_object,
        augmented_metadata,
        max_time_shift=50,
        block_width=250,
        window_padding=100,
        drop_fraction=0.1,
    ):
        """Choose the times at which to sample for a new augmented object.

        Parameters
        ==========
        reference_object : :class:`AstronomicalObject`
            The object to use as a reference for the augmentation.
        augmented_metadata : dict
            The augmented metadata
        max_time_shift : float (optional)
            The new sampling times will be shifted by up to this amount
            relative to the original ones.
        block_width : float (optional)
            A block of observations with a width specified by this parameter
            will be dropped.
        window_padding : float (optional)
            Observations outside of a window bounded by the first and last
            observations in the reference objects light curve with a padding
            specified by this parameter will be dropped.
        drop_fraction : float (optional)
            This fraction of observations will always be dropped when creating
            the augmented light curve.

        Returns
        =======
        sampling_times : pandas Dataframe
            A pandas Dataframe that has the following columns:

            - time : the times of the simulated observations.
            - band : the bands of the simulated observations.
            - reference_time : the times in the reference light curve that
              correspond to the times of the simulated observations.
        """
        reference_observations = reference_object.observations
        sampling_times = reference_observations[["time", "band"]].copy()
        sampling_times["reference_time"] = sampling_times["time"].copy()

        # remove y band observations
        sampling_times = sampling_times[sampling_times["band"] != "lssty"]

        # select active season with target mjd range modeled after SDSS
        lower, upper = 50, 95
        mu, sigma = 88, 10
        duration_model = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        target_duration = int(duration_model.rvs(1))
        midpoint = target_duration // 2

        # choose a random number of days on either side of the peak mjd date
        num_days_on_left = int(np.random.uniform(midpoint - 10, midpoint + 10))
        num_days_on_right = target_duration - num_days_on_left

        # peakmjd = augmented_metadata['true_peakmjd']
        peakmjd = reference_observations["time"].iloc[
            np.argmax(reference_observations["flux"].values)
        ]
        start_time = peakmjd - num_days_on_left
        end_time = peakmjd + num_days_on_right

        # select sampling times
        sampling_times = sampling_times[
            (sampling_times["time"] >= start_time)
            & (sampling_times["time"] <= end_time)
        ]

        if len(sampling_times) == 0:
            return sampling_times

        target_observation_count = self._choose_target_observation_count(augmented_metadata)
        # SDSS images every filter each time it observes, so target observation count = 5 * target observing nights
        target_num_mjds = target_observation_count // 5
        if len(sampling_times) < target_num_mjds:
            # pick some more sampling times within the range
            new_times = np.random.uniform(
                sampling_times["time"].min(),
                sampling_times["time"].max(),
                target_num_mjds - len(sampling_times),
            )

            new_times = pd.DataFrame(
                {
                    "time": new_times,
                    "reference_time": new_times,
                    "band": np.random.choice(sampling_times['band'], size=len(new_times))
                }
            )

            sampling_times = pd.concat([sampling_times, new_times])

        # add each band for every time in sampling_times
        new_bands = defaultdict(list)
        for obs in sampling_times.itertuples():
            unimaged_bands = np.setdiff1d(plasticc_bands, obs.band)
            new_bands['band'].extend(unimaged_bands)
            new_bands['time'].extend([obs.time] * len(unimaged_bands))
            new_bands['reference_time'].extend([obs.reference_time] * len(unimaged_bands))
        sampling_times = pd.concat([sampling_times, pd.DataFrame(new_bands)])

        # drop down to target number of observations or drop random 10%
        num_drop = int(
            max(
                len(sampling_times) - target_observation_count,
                drop_fraction * len(sampling_times),
            )
        )
        drop_indices = np.random.choice(sampling_times.index, num_drop, replace=False)
        sampling_times = sampling_times.drop(drop_indices).copy()

        return sampling_times


    def _choose_target_observation_count(self, augmented_metadata):
        """Choose the target number of observations for a new augmented light
        curve.

        We use a functional form that roughly maps out the number of
        observations in the SDSS dataset.

        Parameters
        ----------
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        target_observation_count : int
            The target number of observations in the new light curve.
        """
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

        return target_observation_count

    def _simulate_light_curve_uncertainties(self, observations, augmented_metadata):
        """Simulate the observation-related noise and detections for a light
        curve.

        For the PLAsTiCC dataset, we estimate the measurement uncertainties for
        each band with a lognormal distribution for both the WFD and DDF
        surveys. Those measurement uncertainties are added to the simulated
        observations.

        Parameters
        ----------
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process. These observations have model flux uncertainties listed
            that should be included in the final uncertainties.
        augmented_metadata : dict
            The augmented metadata

        Returns
        -------
        observations : pandas.DataFrame
            The observations with uncertainties added.
        """
        # Make a copy so that we don't modify the original array.
        observations = observations.copy()

        if len(observations) == 0:
            # No data, skip
            return observations

        # add SDSS-like noise - removing for now, S/N ratios seem close but SDSS flux and err raw numbers are so much higher
        # should be ok since normalization is done, but could do a multiplicative factor, adding a constant is def not the way
        # band_noises = [
        #     (3.7, 0.6, 20),
        #     (2.5, 1, 10),
        #     (2.7, 0.6, 15),
        #     (2.9, 0.6, 27),
        #     (4.4, 0.5, 60)
        # ]

        # for band, (mean, std, offset) in enumerate(band_noises):
        #     band_obs = observations[observations["band"] == band]
        #     noise = np.clip(np.random.lognormal(mean, std, size=len(band_obs)) + offset, 9, None)
        #     flux_jitter = np.random.normal([0]*len(band_obs), 0.1*band_obs['flux_error'].values)

        #     observations.loc[observations.band == plasticc_bands[band], 'flux'] += flux_jitter
        #     observations.loc[observations.band == plasticc_bands[band], 'flux_error'] = np.sqrt(
        #         observations.loc[observations.band == plasticc_bands[band], 'flux_error']**2 + noise**2
        #     )

        return observations

    def _simulate_detection(self, observations, augmented_metadata):
        """Simulate the detection process for a light curve.

        We model the PLAsTiCC detection probabilities with an error function.
        I'm not entirely sure why this isn't deterministic. The full light
        curve is considered to be detected if there are at least 2 individual
        detected observations.

        Parameters
        ==========
        observations : pandas.DataFrame
            The augmented observations that have been sampled from a Gaussian
            Process.
        augmented_metadata : dict
            The augmented metadata

        Returns
        =======
        observations : pandas.DataFrame
            The observations with the detected flag set.
        pass_detection : bool
            Whether or not the full light curve passes the detection thresholds
            used for the full sample.
        """
        s2n = np.abs(observations["flux"]) / observations["flux_error"]
        prob_detected = (erf((s2n - 5.5) / 2) + 1) / 2.0
        observations["detected"] = np.random.rand(len(s2n)) < prob_detected

        # TODO: don't know how to model SDSS detection thresholds, just returning true for now
        pass_detection = np.sum(observations["detected"]) >= 2

        return observations, True

