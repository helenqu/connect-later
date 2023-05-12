from functools import lru_cache
from functools import partial

from typing import Optional, Iterable

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.time_feature import month_of_year, time_features_from_frequency_str
from gluonts.time_feature import TimeFeature
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic, IterableSlice, PseudoShuffled
from gluonts.torch.util import IterableDataset

from torch.utils.data import DataLoader
from transformers import PretrainedConfig

import numpy as np
import pandas as pd
import pdb

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def normalize_data(example):
    # normalize data
    example["target"] = example["target"] / np.max(example["target"])
    return example

def transform_start_field(batch, freq):
    # batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    #TODO: threw out start field, otherwise have to convert from mjd
    batch["start"] = [convert_to_pandas_period("2010-12-29", freq) for date in batch["start"]]
    return batch

def create_transformation(config: PretrainedConfig, time_features: list) -> Transformation:
    # create list of fields to remove later
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            #TODO: shouldn't have any nans, but maybe can use this if we try not doing GP
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,#time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in the life the value of the time series is
            # sort of running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    allow_padding: Optional[bool] = True,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if allow_padding else config.context_length,
            min_future=config.prediction_length,
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(
            min_past=0 if allow_padding else config.context_length,
            min_future=config.prediction_length
        ),
        "test": TestSplitSampler(),
    }[mode]

    print(f"instance splitter created with context length {config.context_length}, lags {config.lags_sequence}")

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader(
    config: PretrainedConfig,
    dataset,
    time_features,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: Optional[bool] = True,
    allow_padding: Optional[bool] = True,
    add_objid: Optional[bool] = False,
    **kwargs,
) -> Iterable:

    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    if config.has_labels:
        TRAINING_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")

    dataset = dataset.map(normalize_data)
    dataset.set_transform(partial(transform_start_field, freq="1M"))

    transformation = create_transformation(config, time_features)
    transformed_data = transformation.apply(dataset, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)


    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train", allow_padding) + SelectFields(
        TRAINING_INPUT_NAMES #+ ["objid"]
    )

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from all the possible transformed time series, 1 in our case)
    # randomly from within the target time series and return an iterator.
    training_instances = instance_splitter.apply(
        Cyclic(transformed_data)
        if shuffle_buffer_length is None
        else PseudoShuffled(
            Cyclic(transformed_data),
            shuffle_buffer_length=shuffle_buffer_length,
        )
    )

    # from the training instances iterator we now return a Dataloader which will
    # continue to sample random windows for as long as it is called
    # to return batch_size of the appropriate tensors ready for training!
    return IterableSlice(
        iter(
            DataLoader(
                IterableDataset(training_instances),
                batch_size=batch_size,
                **kwargs,
            )
        ),
        num_batches_per_epoch,
    )

def create_test_dataloader(
    config: PretrainedConfig,
    dataset,
    time_features,
    batch_size: int,
    allow_padding: Optional[bool] = True,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    if config.has_labels:
        PREDICTION_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")

    dataset = dataset.map(normalize_data)
    dataset.set_transform(partial(transform_start_field, freq="1M"))

    transformation = create_transformation(config, time_features)
    transformed_data = transformation.apply(dataset, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test", allow_padding) + SelectFields(
        PREDICTION_INPUT_NAMES
    )

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    # This returns a Dataloader which will go over the dataset once.
    return DataLoader(
        IterableDataset(testing_instances), batch_size=batch_size, **kwargs
    )
