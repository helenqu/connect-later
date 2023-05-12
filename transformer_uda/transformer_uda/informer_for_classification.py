from typing import List, Optional, Tuple, Union

from transformers import InformerPreTrainedModel, InformerModel
from transformers.models.informer.modeling_informer import InformerConvLayer
from transformers.modeling_outputs import SequenceClassifierOutput

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, functional as F

import pdb

class InformerForSequenceClassification(InformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(f"num labels: {self.num_labels}")
        self.config = config

        self.informer = InformerModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        print(f"classifier dropout: {classifier_dropout}")
        self.conv_layer = InformerConvLayer(config.hidden_size)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.informer(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        encoder_output = outputs.encoder_last_hidden_state

        conv_output = self.conv_layer(encoder_output)
        conv_output = self.conv_layer(conv_output)
        conv_output = self.conv_layer(conv_output)
        conv_output = self.conv_layer(conv_output)
        pooled_output = self.pooler_activation(self.pooler(conv_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss_fn = BCEWithLogitsLoss(weight=weights) if weights is not None else BCEWithLogitsLoss()
        loss = loss_fn(logits, torch.unsqueeze(F.one_hot(labels, num_classes=self.num_labels).float(), 1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
        )
