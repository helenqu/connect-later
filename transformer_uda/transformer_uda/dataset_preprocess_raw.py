"""
preprocess non-GP-interpolated data for use with learnable Fourier PE
"""

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

import torch
from torch.utils.data import DataLoader
from transformers import PretrainedConfig
from datasets import concatenate_datasets

import numpy as np
import pandas as pd
import pdb
from collections import Counter, defaultdict

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def normalize_all(example, field_name):
    # normalize to [0,1] overall (min-max normalization to get rid of negative values)
    values = example[field_name]
    example[field_name] = (values - np.min(values)) / (np.max(values) - np.min(values))
    return example

def normalize_by_channel(example, field_name):
    # normalize to [0,1] by channel (min-max normalization to get rid of negative values)
    for row in range(len(example[field_name])):
        min_value = np.min(example[field_name][row][example[field_name][row] != 0])
        row_values = example[field_name][row]
        example[field_name][row] = (row_values - min_value) / (np.max(row_values) - min_value)
    return example

def create_attention_mask(example):
    # create attention mask to ignore padding
    example["attention_mask"] = np.zeros_like(example["transposed_target"])
    example["attention_mask"][:, example['transposed_times_wv'][0] != 0] = 1 # mask if time value is 0 (padding)
    return example

def mask(example, mask_fraction=0.5, mask_value=0):
    # mask out mask_fraction % of values in the target
    indices_to_replace = np.random.choice(len(example['transposed_target'][0]), int(len(example['transposed_target'][0]) * mask_fraction), replace=False)
    # replace 80% with mask_value, 10% with random value, 10% with original value (Devlin et al. 2018)
    indices_to_mask = np.random.choice(indices_to_replace, int(len(indices_to_replace) * 0.8), replace=False)
    remaining_indices = np.setdiff1d(indices_to_replace, indices_to_mask)
    indices_to_replace_with_random = np.random.choice(remaining_indices, int(len(remaining_indices) * 0.5), replace=False)
    # label for calculating loss: original value for masked, 0 for unmasked (don't want to calculate loss for unmasked)
    unmasked_indices = np.setdiff1d(range(len(example['transposed_target'][0])), indices_to_replace)
    example["mask_label"] = example["transposed_target"]
    example['mask_label'][:, unmasked_indices] = 0

    example['transposed_target'][:, indices_to_mask] = mask_value
    random_indices = np.random.choice(unmasked_indices, len(indices_to_replace_with_random), replace=False)
    example['transposed_target'][:, indices_to_replace_with_random] = example['transposed_target'][:, random_indices]

    return example

def masked_data_collator(mask_probability, data):
    batch = defaultdict(np.array)
    print(len(data), type(data))
    for example in data:
        for key, value in example.items():
            if key == 'objid':
                batch[key].append(value)
            else:
                batch[key] = np.concatenate(batch[key], np.expand_dims(value, axis=0), axis=0)
    return batch
        # labels = np.copy(example['values'])

        # # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # masked_indices = torch.bernoulli(torch.full((1, labels.shape[1]), mask_probability)).bool().squeeze()
        # labels[:, ~masked_indices] = -100  # We only compute loss on masked tokens
        # # print(f"LABELS: {labels}")

        # indices_replaced = torch.bernoulli(torch.full((1, labels.shape[1]), 0.8)).bool().squeeze() & masked_indices

        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full((1, labels.shape[1]), 0.5)).bool().squeeze() & masked_indices & ~indices_replaced
        # # pdb.set_trace()
        # shuffled_indices = torch.randint(len(example['values'][0]), example['values'][0].shape)
        # shuffled_words = example['values'][:, shuffled_indices]
        # # random_word_indices = torch.bernoulli(torch.full((1, labels.shape[1]), mask_probability*0.1)).bool().squeeze()

        # # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        # example['values'][:, indices_replaced] = 0
        # # print(f"VALUES AFTER REPLACING W 0: {example['values']}")
        # example['values'][:, indices_random] = shuffled_words[:, indices_random]
        # # print(f"VALUES AFTER REPLACING W RANDOM: {example['values']}")
        # example['mask_label'] = labels

        # batch.append(example)

    # print(len(batch))
    # return batch

def transform_start_field(batch, freq):
    # batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    #TODO: threw out start field, otherwise have to convert from mjd
    batch["start"] = [convert_to_pandas_period("2010-12-29", freq) for example in batch]
    return batch

def transform_raw_data_example(config, example):
    # was 300 x 2, need to be 2 x 300 (first dim is channel)
    example['transposed_target'] = np.array(example['target']).T
    example['transposed_times_wv'] = np.array(example['times_wv']).T
    # divide by max value to constrain to [0,1]
    example = create_attention_mask(example)
    example = normalize_by_channel(example, "transposed_times_wv")
    example = normalize_all(example, "transposed_target") # normalize flux, flux_err by max overall
    # masking comes after normalization bc mask value is 0
    # if config.mask:
    #     example = mask(example, config.mask_fraction if hasattr(config, "mask_fraction") else 0.5)
    return example

def transform_raw_data(dataset, config: PretrainedConfig):
    # normalize time, data; create attention mask
    dataset = dataset.map(partial(transform_raw_data_example, config))
    dataset.set_transform(partial(transform_start_field, freq="1M"))
    # have to swap out these field names because can't change dataset field shapes in place
    dataset = dataset.remove_columns(["target", "times_wv"])
    dataset = dataset.rename_column("transposed_target", "target")
    dataset = dataset.rename_column("transposed_times_wv", "times_wv")

    # remove/rename fields
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)
    remove_field_names.append('start')

    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names),
        AsNumpyArray(
            field="times_wv",
            # we expect an extra dim for the multivariate case:
            expected_ndim=2,
        ),
        AsNumpyArray(
            field=FieldName.TARGET,
            # we expect an extra dim for the multivariate case:
            expected_ndim=2,
        ),
        AsNumpyArray(
            field='attention_mask',
            # we expect an extra dim for the multivariate case:
            expected_ndim=2,
        ),
        RenameFields(
            mapping={
                FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                FieldName.FEAT_STATIC_REAL: "static_real_features",
                "times_wv": "time_features",
                FieldName.TARGET: "values",
                "attention_mask": "observed_mask",
            }
        )]
    ).apply(dataset)

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

def create_train_dataloader_raw(
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

    if hasattr(config, "balance"):
        print("balancing training data")
        abundances = Counter(dataset['labels'])
        max_num = max(abundances.values())
        for i in range(config.num_labels):
            if abundances[i] == 0:
                continue
            tmp_dataset = dataset.filter(lambda x: x["labels"] == i)
            dataset_for_type = dataset.filter(lambda x: x["labels"] == i)
            num_multiples = ((max_num - abundances[i]) // abundances[i]) + 1
            for j in range(num_multiples):
                tmp_dataset = concatenate_datasets([tmp_dataset, dataset_for_type])
            print(f"dataset for type {i} has {len(tmp_dataset)} examples")
            tmp_dataset = tmp_dataset.select(range(max_num - abundances[i]))
            dataset = concatenate_datasets([dataset, tmp_dataset])
        print("balanced training data: {Counter(dataset['labels'])}")

    transformed_data = transform_raw_data(dataset, config)
    # if cache_data:
    #     transformed_data = Cached(transformed_data)

    if config.mask:
        return DataLoader(
            IterableDataset(transformed_data),
            batch_size=batch_size,
            collate_fn=partial(masked_data_collator, config.mask_probability)
        )

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

def create_test_dataloader_raw(
    config: PretrainedConfig,
    dataset,
    time_features,
    batch_size: int,
    allow_padding: Optional[bool] = True,
    add_objid: Optional[bool] = False,
    compute_loss: Optional[bool] = False,
    shuffle_buffer_length: Optional[int] = None,
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

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    if compute_loss:
        PREDICTION_INPUT_NAMES += [
            "future_values",
            "future_bserved_mask",
        ]

    transformed_data = transform_raw_data(dataset, config)
    # still passing in 'train' because otherwise it will take the last data point (no room for future values)
    instance_sampler = create_instance_splitter(config, "train", allow_padding) + SelectFields(
        PREDICTION_INPUT_NAMES
    )

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(
        transformed_data,
        # if shuffle_buffer_length is None
        # else PseudoShuffled(
        #     transformed_data,
        #     shuffle_buffer_length=shuffle_buffer_length,
        # ),
        is_train=False
    )

    # This returns a Dataloader which will go over the dataset once.
    return DataLoader(
        IterableDataset(testing_instances), batch_size=batch_size, **kwargs
    )

