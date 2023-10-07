from gluonts.transform import (
    AddObservedValuesIndicator,
    AsNumpyArray,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RenameFields,
    ValidationSplitSampler,
    TestSplitSampler,
    Transformation,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic, IterableSlice, PseudoShuffled
from gluonts.torch.util import IterableDataset

import pandas as pd
import numpy as np
from functools import partial
import pdb

from datasets import Dataset

def create_instance_splitter(
    mode: str,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    max_lags = 50
    prediction_length = 10
    context_length = 2*prediction_length

    instance_sampler = {
        "train": ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        ),
        "validation": ValidationSplitSampler(min_future=prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="target",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=context_length + max_lags, #config.context_length + max(config.lags_sequence),
        future_length=prediction_length,
        time_series_fields=[]#["time_features", "observed_mask"],
    )


def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)


def transform_start_field(batch, freq):
    batch["start"] = np.array([convert_to_pandas_period(date, freq) for date in batch["start"]])
    return batch

def create_dataset() -> list:
    ts_length = 100
    num_ts = 100
    ds = []
    for i in range(num_ts):
        ds.append(
            {
                "target": np.array(list(range(i*ts_length, (i+1)*ts_length))),
                "start": "01-01-2019",
                "item_id": str(i),
            }
        )
    # start = [pd.Period("01-01-2019", freq="1H")] * num_ts
    # target = [range(i*ts_length, (i+1)*ts_length) for i in range(num_ts)]
    # return {'start': start, 'target': target}
    return Dataset.from_pandas(pd.DataFrame(data=ds))

example_ds = create_dataset()
example_ds.set_transform(partial(transform_start_field, freq="1H"))
transformations = AsNumpyArray(
                    field='target',
                    expected_ndim=1
                ) + AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ) #+ RenameFields(
                    # mapping={
                    #     FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    #     FieldName.FEAT_STATIC_REAL: "static_real_features",
                    #     FieldName.FEAT_TIME: "time_features",
                    #     FieldName.TARGET: "values",
                    #     FieldName.OBSERVED_VALUES: "observed_mask",
                    # }
                # )
transformed_ds = transformations.apply(example_ds)
training_instances = create_instance_splitter("train").apply(PseudoShuffled(Cyclic(transformed_ds), shuffle_buffer_length=100))
pdb.set_trace()
print(next(iter(training_instances)))

