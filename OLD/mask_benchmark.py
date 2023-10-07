from transformer_uda.huggingface_informer import get_dataset
from transformer_uda.dataset_preprocess_raw import create_train_dataloader_raw

from transformers import InformerConfig, InformerForPrediction, PretrainedConfig
import time

if __name__ == '__main__':

    dataset = get_dataset("/pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples/")#, data_subset_file="/pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples/just_train_set.txt")
    model_config = InformerConfig(
        # in the multivariate setting, input_size is the number of variates in the time series per time step
        input_size=2,
        # prediction length:
        prediction_length=10,
        # context length:
        context_length=300,
        # lags value copied from 1 week before:
        lags_sequence=[0],
        # we'll add 5 time features ("hour_of_day", ..., and "age"):
        num_time_features=2, #wavelength + time
        # informer params:
        dropout=0.5,
        encoder_layers=8,
        decoder_layers=8,
        # project input from num_of_variates*len(lags_sequence)+num_time_features to:
        d_model=768,
        scaling=None,
        has_labels=False,
        mask=True,
        mask_probability=0.5,
    )
    train_dataloader = create_train_dataloader_raw(
        config=model_config,
        dataset=dataset['train'],
        time_features=None, # not used in raw version
        batch_size=1024,
        num_batches_per_epoch=1000,
        shuffle_buffer_length=100_000,
        allow_padding=True,
        cache_data=False
    )

    times = []
    start = time.time()
    for idx, batch in enumerate(train_dataloader):
        if idx > 100:
            break
        end = time.time()
        times.append(end - start)
        start = time.time()
    print(f"single batch took an average of {sum(times) / len(times)} seconds")
    print(f"All times: {times}")
