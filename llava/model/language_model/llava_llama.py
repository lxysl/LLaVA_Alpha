#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.alpha_token_idx = None  # set in LlavaMetaForCausalLM.initialize_alpha_tokenizer()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if self.model.config.alpha:
            input_ids_backup = input_ids
            position_ids_backup = position_ids
            attention_mask_backup = attention_mask
            past_key_values_backup = past_key_values
            labels_backup = labels
            (
                input_ids,
                position_ids,  # None
                attention_mask,
                past_key_values,  # None
                inputs_embeds,
                labels,
                alpha_token_mask,  # (batch_size, seq_len)
                image_embeds,  # (batch_size, patch_size, patch_size, hidden_size)
            ) = self.prepare_inputs_labels_for_multimodal_alpha(
                input_ids,  # given
                position_ids,
                attention_mask,  # given
                past_key_values,
                labels,  # given
                images,  # given
                image_sizes,
                None
            )
            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict
            )
            last_hidden_state = output.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
            alpha_hidden_states = last_hidden_state[alpha_token_mask]  # (num_alpha, hidden_size)
            alpha_projections = self.get_model().state_projector(alpha_hidden_states)  # (num_alpha, projection_size)

            assert alpha_projections.size(0) == alpha_token_mask.sum() == len(images), \
                f"alpha_projections.size={alpha_projections.size}, " \
                f"alpha_token_mask.sum()={alpha_token_mask.sum()}, " \
                f"len(images)={len(images)}"

            (sparse_embeddings, dense_embeddings) = self.get_model().prompt_encoder(points=None,
                                                                                    boxes=None,
                                                                                    masks=None,
                                                                                    text_embeds=alpha_projections.unsqueeze(1))
            sparse_embeddings = sparse_embeddings.to(alpha_projections.dtype)

            low_res_masks, _ = self.get_model().alpha_decoder(image_embeddings=self.get_model().out_projector(image_embeds).permute(0, 3, 1, 2),  # (bs, 256, 16, 16)
                                                              image_pe=self.get_model().prompt_encoder.get_dense_pe(),  # (1, 256, 16, 16)
                                                              sparse_prompt_embeddings=sparse_embeddings,  # (bs, 1, 256)
                                                              dense_prompt_embeddings=dense_embeddings,  # (bs, 256, 16, 16)
                                                              multimask_output=False)  # (bs, 4, 64, 64), (bs, 4)
            alpha_mask = F.interpolate(low_res_masks, self.get_model().prompt_encoder.input_image_size, mode='bilinear', align_corners=False)[:, [0], :, :]  # (bs, 1, 224, 224)

            (
                input_ids,
                position_ids,  # None
                attention_mask,
                past_key_values,  # None
                inputs_embeds,
                labels,
                alpha_token_mask,  # (batch_size, seq_len)
                image_embeds,  # (batch_size, patch_size, patch_size, hidden_size)
            ) = self.prepare_inputs_labels_for_multimodal_alpha(
                input_ids_backup,  # given
                position_ids_backup,
                attention_mask_backup,  # given
                past_key_values_backup,
                labels_backup,  # given
                images,  # given
                image_sizes,
                alpha_mask
            )
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        else:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,  # None
                    attention_mask,
                    past_key_values,  # None
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,  # given
                    position_ids,
                    attention_mask,  # given
                    past_key_values,
                    labels,  # given
                    images,  # given
                    image_sizes
                )

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
