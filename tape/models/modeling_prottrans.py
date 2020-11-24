"""PyTorch BERT BFD model from ProtTrans. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .modeling_utils import ProteinModel
from .modeling_utils import ProteinConfig
from .modeling_utils import MLMHead
from .modeling_utils import ValuePredictionHead
from .modeling_utils import SequenceClassificationHead
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import PairwiseContactPredictionHead
from ..registry import registry

# ProtTrans transformers (huggingface) dependency
from transformers import BertModel

logger = logging.getLogger(__name__)


class ProteinBFDConfig(ProteinConfig):
    r"""
        :class:`~pytorch_transformers.ProteinBertConfig` is the configuration class to store the
        configuration of a `ProteinBertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `ProteinBertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the ProteinBert encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the ProteinBert encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the ProteinBert encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `ProteinBertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """

    def __init__(self,
                 vocab_size: int = 30,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 30,
                 num_attention_heads: int = 16,
                 intermediate_size: int = 4096,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 40000,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 trainable_encoder: bool = False,  # Disabled by default
                 **kwargs):

        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.trainable_encoder = trainable_encoder


class ProteinBFDAbstractModel(ProteinModel):
    config_class = ProteinBFDConfig

    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        pass


@registry.register_task_model('embed', 'prottrans')
class ProteinBFDModel(ProteinBFDAbstractModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")

    def forward(self, input_ids, input_mask=None):
        h = self.model(input_ids)
        return h


@registry.register_task_model('masked_language_modeling', 'prottrans')
class ProteinBertForMaskedLM(ProteinBFDAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBFDModel(config)
        self.mlm = MLMHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('fluorescence', 'prottrans')
@registry.register_task_model('stability', 'prottrans')
@registry.register_task_model('melting_point_regression', 'prottrans')
@registry.register_task_model('fireprot', 'prottrans')
class ProteinBertForValuePrediction(ProteinBFDAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBFDModel(config)
        self.predict = ValuePredictionHead(config.hidden_size)

        if not config.trainable_encoder:
            for name, param in self.bert.named_parameters():
                # Make sure the pooler can keep learning
                if "pooler" not in name:
                    param.requires_grad = False

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('remote_homology', 'prottrans')
@registry.register_task_model('melting_point_classification', 'prottrans')
class ProteinBertForSequenceClassification(ProteinBFDAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBFDModel(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        if not config.trainable_encoder:
            for name, param in self.bert.named_parameters():
                # Make sure the pooler can keep learning
                if "pooler" not in name:
                    param.requires_grad = False

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('secondary_structure', 'prottrans')
class ProteinBertForSequenceToSequenceClassification(ProteinBFDAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBFDModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('contact_prediction', 'prottrans')
class ProteinBertForContactPrediction(ProteinBFDAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBFDModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
