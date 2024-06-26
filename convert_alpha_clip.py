import os
import shutil
import argparse
import collections

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor


class AlphaCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = getattr(args, 'vision_tower', None)
        self.vision_tower_path = getattr(args, 'vision_tower_path', None)
        self.select_layer = -2
        self.select_feature = 'patch'
        self.alpha_clip_weight_path = getattr(args, 'alpha_clip_weight_path', None)
        self.save_path = getattr(args, 'converted_alpha_clip_weight_path', None)

        self.load_model()

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

    def save_converted_weights(self):
        assert os.path.exists(self.vision_tower_path), f'{self.vision_tower_path} not exists'

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # copy configs in huggingface CLIP model folder to the save_path
        for filename in os.listdir(self.vision_tower_path):
            source_file = os.path.join(self.vision_tower_path, filename)
            if filename != 'pytorch_model.bin' and os.path.isfile(source_file):
                destination_file = os.path.join(self.save_path, filename)
                shutil.copy2(source_file, destination_file)

        model_state_dict = self.vision_tower.state_dict()
        save_path = os.path.join(self.save_path, "pytorch_model.bin")
        torch.save(model_state_dict, save_path)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_tower', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--vision_tower_path', type=str, default='~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14')
    parser.add_argument('--alpha_clip_weight_path', type=str, default='./checkpoints/clip_l14_336_grit_20m_4xe.pth')
    parser.add_argument('--converted_alpha_clip_weight_path', type=str, default='./checkpoints/alpha_clip_l14_336_grit_20m_4xe')
    args = parser.parse_args()

    vision_tower = AlphaCLIPVisionTower(args)
    vision_tower.save_converted_weights()
