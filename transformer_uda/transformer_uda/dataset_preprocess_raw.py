"""
preprocess non-GP-interpolated data for use with learnable Fourier PE
"""

from functools import lru_cache
from functools import partial

from typing import Optional, Iterable

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
from transformers.models.informer.modeling_informer import InformerMeanScaler, InformerStdScaler, InformerNOPScaler
from datasets import concatenate_datasets

import numpy as np
import pandas as pd
import pdb
from collections import Counter

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

def masked_data_collator(mask_probability, cols_to_keep, data):
    batch = {}
    # defaultdict(partial(np.ndarray, 0))
    for key in data[0].keys():
        batch_key = key if key not in ['values', 'observed_mask', 'time_features'] else f"past_{key}"
        if batch_key not in cols_to_keep:
            continue
        batch[batch_key] = torch.stack([torch.tensor(example[key]) for example in data])

    labels = batch['past_values'][:, 0, :].clone() # only take flux values, should be [batch_size, 1, seq_len]

    masked_indices = torch.bernoulli(torch.full(labels.shape, mask_probability)).bool().squeeze()
    labels[~masked_indices] = 0  # We only compute loss on masked tokens
    # print(f"LABELS: {labels}")

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().squeeze() & masked_indices

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().squeeze() & masked_indices & ~indices_replaced
    # shuffled_indices = torch.randint(len(batch['values']), batch['values'].shape)
    # shuffled_words = batch['values'][shuffled_indices, :, :]

    indices_replaced = torch.tile(torch.unsqueeze(indices_replaced, 1), (1, 2, 1))
    indices_random = torch.tile(torch.unsqueeze(indices_random, 1), (1, 2, 1))
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    batch['past_values'][indices_replaced] = 0
    batch['past_values'][indices_random] = torch.rand(batch['past_values'][indices_random].shape)
    batch['past_values'] = torch.transpose(batch['past_values'], 1, 2)
    batch['past_time_features'] = torch.transpose(batch['past_time_features'], 1, 2)
    #shuffled_words[indices_random]

    if 'mask_label' in cols_to_keep:
        batch['mask_label'] = labels

    return batch

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
    if not config.mask:
        dataset = dataset.add_column("start", [0] * len(dataset))
        dataset.set_transform(partial(transform_start_field, freq="1M"))
    # have to swap out these field names because can't change dataset field shapes in place
    dataset = dataset.remove_columns(["target", "times_wv"])
    dataset = dataset.rename_column("transposed_target", "target")
    dataset = dataset.rename_column("transposed_times_wv", "times_wv")

    # remove/rename fields
    name_mapping = {
                "times_wv": "time_features",
                FieldName.TARGET: "values",
                "attention_mask": "observed_mask",
            }

    # dataset['times_wv'] = np.asarray(dataset['times_wv'])
    # dataset['target'] = np.asarray(dataset['target'])
    # dataset['attention_mask'] = np.asarray(dataset['attention_mask'])
    dataset = dataset.rename_columns(name_mapping)
    dataset = dataset.with_format('np')

    return dataset

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
        if "redshift" in dataset.column_names:
            dataset = dataset.rename_column("redshift", "static_real_features")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    if config.has_labels:
        TRAINING_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")
    elif config.mask:
        TRAINING_INPUT_NAMES.append("mask_label")

    if hasattr(config, "balance") and config.balance:
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

    # transformed_data = transformed_data.to_iterable_dataset(num_shards=128).shuffle(buffer_size=shuffle_buffer_length)
    # if cache_data:
    #     transformed_data = Cached(transformed_data)

    # sampler = RandomSampler(transformed_data)

    if config.mask:
        transformed_data = transformed_data.shuffle(seed=111).flatten_indices()  # TODO add seed to args
        mask_probability = 0. if config.has_labels else 0.8 # don't mask for fine-tuning
        # print("debug: selecting 50 examples")
        # transformed_data = transformed_data.select(range(50))
        return DataLoader(
            transformed_data,
            batch_size=batch_size,
            # sampler=sampler,
            num_workers=1,
            collate_fn=partial(masked_data_collator, mask_probability, TRAINING_INPUT_NAMES),
            # shuffle=True
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
        if "redshift" in dataset.column_names:
            dataset = dataset.rename_column("redshift", "static_real_features")

    if config.has_labels:
        PREDICTION_INPUT_NAMES.append("labels")
        dataset = dataset.rename_column("label", "labels")

    if add_objid:
        PREDICTION_INPUT_NAMES.append("objid")

    if compute_loss:
        PREDICTION_INPUT_NAMES += [
            "future_values",
            "future_observed_mask",
        ]

    transformed_data = transform_raw_data(dataset, config)
    if config.mask:
        transformed_data = transformed_data.shuffle(seed=111).flatten_indices()  # TODO add seed to args
        mask_probability = 0. if config.has_labels else 0.8 # don't mask for fine-tuning
        return DataLoader(
            transformed_data,
            batch_size=batch_size,
            # sampler=sampler,
            num_workers=1,
            collate_fn=partial(masked_data_collator, mask_probability, PREDICTION_INPUT_NAMES)
        )
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

def create_network_inputs(
    config: PretrainedConfig,
    past_values: torch.Tensor,
    past_time_features: torch.Tensor,
    static_categorical_features: Optional[torch.Tensor] = None,
    static_real_features: Optional[torch.Tensor] = None,
    past_observed_mask: Optional[torch.Tensor] = None,
    future_values: Optional[torch.Tensor] = None,
    future_time_features: Optional[torch.Tensor] = None,
):
    if config.scaling == "mean" or config.scaling is True:
        scaler = InformerMeanScaler(dim=1, keepdim=True)
    elif config.scaling == "std":
        scaler = InformerStdScaler(dim=1, keepdim=True)
    else:
        scaler = InformerNOPScaler(dim=1, keepdim=True)

    past_length = config.context_length + max(config.lags_sequence)
    # time feature
    time_feat = (
        torch.cat(
            (
                past_time_features[:, past_length - config.context_length :, ...],
                future_time_features,
            ),
            dim=1,
        )
        if future_values is not None
        else past_time_features[:, past_length - config.context_length :, ...]
    )

    # target
    if past_observed_mask is None:
        past_observed_mask = torch.ones_like(past_values)

    context = past_values[:, -config.context_length :]
    observed_context = past_observed_mask[:, -config.context_length :]
    _, loc, scale = scaler(context, observed_context)

    inputs = (
        (torch.cat((past_values, future_values), dim=1) - loc) / scale
        if future_values is not None
        else (past_values - loc) / scale
    )

    # static features
    log_abs_loc = loc.abs().log1p() if config.input_size == 1 else loc.squeeze(1).abs().log1p()
    log_scale = scale.log() if config.input_size == 1 else scale.squeeze(1).log()
    static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

    if static_real_features is not None:
        static_feat = torch.cat((static_real_features.unsqueeze(1), static_feat), dim=1)
    if static_categorical_features is not None:
        embedded_cat = embedder(static_categorical_features)
        static_feat = torch.cat((embedded_cat, static_feat), dim=1)
    expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)

    # all features
    features = torch.cat((expanded_static_feat, time_feat), dim=-1)

    # lagged features
    subsequences_length = (
        config.context_length + config.prediction_length
        if future_values is not None
        else config.context_length
    )
    lagged_sequence = get_lagged_subsequences(config=config, sequence=inputs, subsequences_length=subsequences_length)
    lags_shape = lagged_sequence.shape
    reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)

    if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
        raise ValueError(
            f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
        )

    # transformer inputs
    transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)

    return transformer_inputs, loc, scale, static_feat

def get_lagged_subsequences(
    config: PretrainedConfig, sequence: torch.Tensor, subsequences_length: int, shift: int = 0
) -> torch.Tensor:
    sequence_length = sequence.shape[1]
    indices = [lag - shift for lag in config.lags_sequence]

    if max(indices) + subsequences_length > sequence_length:
        raise ValueError(
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )

    lagged_values = []
    for lag_index in indices:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        lagged_values.append(sequence[:, begin_index:end_index, ...])
    return torch.stack(lagged_values, dim=-1)
