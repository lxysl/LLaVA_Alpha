import types
import collections
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling


def rewritten_embedding_forward(self, pixel_values: torch.FloatTensor, alpha_values: torch.FloatTensor) -> torch.Tensor:
    # print("[Warning] using rewrited alpha forword")
    assert pixel_values.shape[0] == alpha_values.shape[0], "pixel_values and alpha should have the same batch size"
    # pixel_values.shape = [batch_size, 3, height, width], alpha.shape = [batch_size, 1, height, width]
    batch_size = pixel_values.shape[0]
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [batch_size, width, grid, grid]
    patch_embeds = patch_embeds + self.patch_embedding_alpha(alpha_values)
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


def rewritten_vision_transformer_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    alpha_values: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")
    if alpha_values is None:
        raise ValueError("You have to specify alpha")

    hidden_states = self.embeddings(pixel_values, alpha_values)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    pooled_output = last_hidden_state[:, 0, :]
    pooled_output = self.post_layernorm(pooled_output)

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def rewritten_vision_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    alpha_values: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    Examples:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, CLIPVisionModel

    >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled CLS states
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    return self.vision_model(
        pixel_values=pixel_values,
        alpha_values=alpha_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


class AlphaCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.alpha_clip_weight_path = getattr(args, 'alpha_clip_weight_path', None)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        visual_encoder = self.vision_tower.vision_model
        visual_encoder.embeddings.patch_embedding_alpha = torch.nn.Conv2d(in_channels=1,
                                                                          out_channels=visual_encoder.embeddings.patch_embedding.out_channels, 
                                                                          kernel_size=visual_encoder.embeddings.patch_embedding.kernel_size, 
                                                                          stride=visual_encoder.embeddings.patch_embedding.stride, 
                                                                          bias=False)
        visual_encoder.embeddings.forward = types.MethodType(rewritten_embedding_forward, visual_encoder.embeddings)
        visual_encoder.forward = types.MethodType(rewritten_vision_transformer_forward, visual_encoder)
        self.vision_tower.forward = types.MethodType(rewritten_vision_model_forward, self.vision_tower)

        state_dict = torch.load(self.alpha_clip_weight_path, map_location=device_map)
        converted_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if 'transformer.resblocks' in k:
                new_key = k.replace('transformer.resblocks', 'encoder.layers').replace('attn', 'self_attn').replace('ln_1', 'layer_norm1').replace('ln_2', 'layer_norm2') \
                            .replace('c_fc', 'fc1').replace('c_proj', 'fc2')
                if ('self_attn' in new_key) and ('out' not in new_key): # split qkv attn
                    if 'weight' in new_key :
                        converted_dict[new_key.replace('in_proj', 'q_proj')] = v[:1024, :]
                        converted_dict[new_key.replace('in_proj', 'k_proj')] = v[1024:2048, :]
                        converted_dict[new_key.replace('in_proj', 'v_proj')] = v[2048:, :]
                    else:
                        assert 'bias' in new_key
                        converted_dict[new_key.replace('in_proj', 'q_proj')] = v[:1024]
                        converted_dict[new_key.replace('in_proj', 'k_proj')] = v[1024:2048]
                        converted_dict[new_key.replace('in_proj', 'v_proj')] = v[2048:]
                else:
                    converted_dict[new_key] = v
            else:
                new_key = k.replace('class_embedding', 'embeddings.class_embedding') \
                            .replace('conv1.weight', 'embeddings.patch_embedding.weight') \
                            .replace('positional_embedding', 'embeddings.position_embedding.weight') \
                            .replace('conv1_alpha.weight', 'embeddings.patch_embedding_alpha.weight') \
                            .replace('ln_pre.weight', 'pre_layrnorm.weight') \
                            .replace('ln_pre.bias', 'pre_layrnorm.bias') \
                            .replace('ln_post.weight', 'post_layernorm.weight') \
                            .replace('ln_post.bias', 'post_layernorm.bias')
                converted_dict[new_key] = v
        visual_encoder.load_state_dict(converted_dict, strict=False)

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # (batch_size, num_patches, hidden_size)
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, alphas):
        if type(images) is list:
            image_features = []
            for image, alpha in zip(images, alphas):
                image, alpha = image.to(device=self.device, dtype=self.dtype).unsqueeze(0), alpha.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                image_forward_out = self.vision_tower(pixel_values=image, alpha_values=alpha, output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            images, alphas = images.to(device=self.device, dtype=self.dtype), alphas.to(device=self.device, dtype=self.dtype)
            image_forward_outs = self.vision_tower(pixel_values=images, alpha_values=alphas, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
