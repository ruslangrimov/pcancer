import types

import torch
from torch import nn

from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.unet.decoder import DecoderBlock


def get_model(classes, decoder=True, labels=True):
    class AutoDecoder(nn.Module):
        def __init__(
                self,
                channels,
                use_batchnorm=True,
                attention_type=None,
        ):
            super().__init__()

            in_channels = channels[:0:-1]
            out_channels = channels[-2::-1]

            kwargs = dict(use_batchnorm=use_batchnorm,
                          attention_type=attention_type)
            blocks = [
                DecoderBlock(in_ch, 0, out_ch, **kwargs)
                for in_ch, out_ch in zip(in_channels, out_channels)
            ]
            self.blocks = nn.ModuleList(blocks)

        def forward(self, features):
            x = features

            for i, decoder_block in enumerate(self.blocks):
                x = decoder_block(x)

            x = torch.sigmoid(x)
            # x = torch.tanh(x)

            return x

    def forward(self, x, return_features=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        out = (masks,)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            out = out + (labels,)

        if self.autodecoder is not None:
            decoded = self.autodecoder(features[-1])
            out = out + (decoded,)

        return ((features[-1],) if return_features else ()) + out

    model = Unet('resnet18', encoder_weights=None,
                 activation=None,
                 classes=classes,
                 aux_params={
                     'classes': classes,
                     'activation': None} if labels else None)

    model.forward = types.MethodType(forward, model)
    channels = model.encoder.out_channels
    if decoder:
        model.autodecoder = AutoDecoder(channels)
    else:
        model.autodecoder = None

    return model
