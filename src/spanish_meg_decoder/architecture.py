"""Brain2Qwerty ConvConformer architecture adapted for standalone inference.

Copyright (c) Meta Platforms, Inc. and affiliates.
Upstream: https://github.com/facebookresearch/brain2qwerty
License: CC BY-NC 4.0
Modification: the published encoder configuration and model are colocated here.
"""

from __future__ import annotations

import torch
from torch import nn

from neuraltrain.models.conv_transformer import ConvTransformer, ConvTransformerModel


class ConvConformer(ConvTransformer):
    aux_prediction: bool = False

    def build(
        self, n_in_channels: int, n_outputs: int | None = None
    ) -> "ConvConformerModel":
        return ConvConformerModel(
            n_in_channels, n_outputs or self.output_layer_dim, config=self
        )


class ConvConformerModel(ConvTransformerModel):
    def __init__(
        self, in_channels: int, out_channels: int | None, config: ConvConformer
    ) -> None:
        super().__init__(in_channels, out_channels, config)
        self.aux_prediction = config.aux_prediction
        if config.aux_prediction:
            self.intermediate_linear = nn.Linear(out_channels or self.dim, self.dim)
            self.shared_layer_norm = nn.LayerNorm(self.dim)

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
        neuro_device_type: str | None = None,
    ) -> dict[str, torch.Tensor]:
        x = x.transpose(1, 2)
        z = self._encoder_and_downsampling_forward(
            x, subject_ids=day_idx, channel_positions=channel_positions
        )
        z_enc = z
        z_aux = None
        if self.aux_prediction:
            z = self.shared_layer_norm(z)
            z_aux = self.output_layer(z)
            z = z + self.intermediate_linear(torch.softmax(z_aux, dim=-1))

        c_in = self._pre_transformer_forward(z, neuro_device_type=neuro_device_type)
        z_final = self.transformer(c_in)
        c_out = z_final.mean(dim=1) if self.output_avg_pool else z_final
        if self.aux_prediction:
            c_out = self.shared_layer_norm(c_out)
        c_out = self.output_layer(c_out)
        output = {
            "z": z_aux if self.aux_prediction else z_enc,
            "z_enc": z_enc,
            "z_final": z_final,
            "c_out": c_out,
        }
        if z_aux is not None:
            output["z_aux"] = z_aux
        return output


def build_model(n_in_channels: int = 306, n_outputs: int = 29) -> nn.Module:
    config = ConvConformer(
        dim=1024,
        encoder_config={
            "name": "SimpleConv",
            "dropout_input": 0.2,
            "conv_dropout": 0.5,
            "hidden": 1500,
            "batch_norm": True,
            "depth": 4,
            "dilation_period": 3,
            "kernel_size": 5,
            "relu_leakiness": 0.01,
            "initial_linear": 512,
            "gelu": True,
            "skip": True,
            "scale": 0.1,
            "subject_layers_config": {},
            "merger_config": {
                "n_virtual_channels": 270,
                "fourier_emb_config": {
                    "n_freqs": None,
                    "total_dim": 2048,
                    "n_dims": 2,
                },
                "dropout": 0.2,
                "usage_penalty": 1.0,
                "per_subject": True,
                "embed_ref": False,
            },
        },
        transformer_config={
            "name": "Conformer",
            "ffn_dim": 1024,
            "num_heads": 4,
            "num_layers": 4,
            "depthwise_conv_kernel_size": 17,
            "dropout": 0.3,
            "use_group_norm": True,
            "convolution_first": False,
        },
        temporal_downsampling_config={"kernel_size": 16, "stride": 4},
        aux_prediction=True,
    )
    return config.build(n_in_channels=n_in_channels, n_outputs=n_outputs)

