import os
from torch import nn
from .transformer import TwoWayTransformer
from .prompt_encoder import PromptEncoder
from .alpha_decoder import AlphaDecoder


def build_state_projector(config, **kwargs):
    in_dim = getattr(config, 'text_hidden_size', 4096)
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(inplace=True),
        nn.Linear(in_dim, 256),
        nn.Dropout(0.0),
    )


def build_vision_projector_out(config, **kwargs):
    image_feature_hidden_size = getattr(config, 'image_feature_hidden_size', 1024)
    return nn.Linear(image_feature_hidden_size, 256, bias=False)


def build_prompt_encoder(config, **kwargs):
    embed_dim = 256
    image_embedding_size = getattr(config, 'image_embedding_size', (16, 16))
    input_image_size = getattr(config, 'input_image_size', (224, 224))

    return PromptEncoder(embed_dim=embed_dim,
                         image_embedding_size=image_embedding_size,
                         input_image_size=input_image_size,
                         mask_in_chans=16)


def build_alpha_decoder(config, **kwargs):
    transformer_dim = 256
    transformer = TwoWayTransformer(depth=2,
                                    embedding_dim=transformer_dim,
                                    mlp_dim=2048,
                                    num_heads=8,)

    return AlphaDecoder(transformer_dim=transformer_dim,
                        transformer=transformer)
