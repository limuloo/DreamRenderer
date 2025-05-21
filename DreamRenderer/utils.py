import argparse
import numpy as np
import torch
import os
import yaml
import random
from diffusers.utils.import_utils import is_accelerate_available
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import EulerDiscreteScheduler
import cv2
if is_accelerate_available():
    from accelerate import init_empty_weights
from contextlib import nullcontext


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


import torch
from typing import Callable, Dict, List, Optional, Union
from collections import defaultdict


def get_all_processor_keys(model, parent_name=''):
    all_processor_keys = []
    
    for name, module in model.named_children():
        full_name = f'{parent_name}.{name}' if parent_name else name
        
        # Check if the module has 'processor' attribute
        if hasattr(module, 'processor'):
            all_processor_keys.append(f'{full_name}.processor')
        
        # Recursively check submodules
        all_processor_keys.extend(get_all_processor_keys(module, full_name))
    
    return all_processor_keys

from diffusers.models.controlnets.controlnet import BaseOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from dataclasses import dataclass
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
@dataclass
class SD3ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]

from diffusers.models.modeling_outputs import Transformer2DModelOutput
def new_forward_SD3_CN(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # if self.pos_embed is not None and hidden_states.ndim != 4:
        #     raise ValueError("hidden_states must be 4D when pos_embed is used")

        # # SD3.5 8b controlnet does not have a `pos_embed`,
        # # it use the `pos_embed` from the transformer to process input before passing to controlnet
        # elif self.pos_embed is None and hidden_states.ndim != 3:
        #     raise ValueError("hidden_states must be 3D when pos_embed is not used")

        # if self.context_embedder is not None and encoder_hidden_states is None:
        #     raise ValueError("encoder_hidden_states must be provided when context_embedder is used")
        # # SD3.5 8b controlnet does not have a `context_embedder`, it does not use `encoder_hidden_states`
        # elif self.context_embedder is None and encoder_hidden_states is not None:
        #     raise ValueError("encoder_hidden_states should not be provided when context_embedder is not used")

        # import pdb
        # pdb.set_trace()

        # 如果hidden_states是 list
        if isinstance(hidden_states, list):
            feature_list = []
            bridge_image_token_index_list = []
            begin_index = 0
            for latents_index, feature in enumerate(hidden_states):
                assert isinstance(controlnet_cond, list)
                control_feature = controlnet_cond[latents_index]
                feature = self.pos_embed(feature)
                feature = feature + self.pos_embed_input(control_feature)
                feature_list.append(feature)
                bridge_image_token_index_list.append(list(range(begin_index, begin_index + feature.shape[1])))
                begin_index = begin_index + feature.shape[1]
            hidden_states = torch.cat(feature_list, dim=1)
            joint_attention_kwargs['bridge_image_token_index_list'] = bridge_image_token_index_list
        else:
            if self.pos_embed is not None:  # torch.Size([1, 16, 128, 128])
                hidden_states = self.pos_embed(hidden_states)  # torch.Size([1, 4096, 1536]), takes care of adding positional embeddings too.
            # add
            hidden_states = hidden_states + self.pos_embed_input(controlnet_cond)

        temb = self.time_text_embed(timestep, pooled_projections)

        if self.context_embedder is not None:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        

        block_res_samples = ()

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if self.context_embedder is not None:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                    )
                else:
                    # SD3.5 8b controlnet use single transformer block, which does not use `encoder_hidden_states`
                    hidden_states = self._gradient_checkpointing_func(block, hidden_states, temb)

            else:
                if self.context_embedder is not None:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                        joint_attention_kwargs = joint_attention_kwargs
                    )
                else:
                    # SD3.5 8b controlnet use single transformer block, which does not use `encoder_hidden_states`
                    hidden_states = block(hidden_states, temb)

            block_res_samples = block_res_samples + (hidden_states,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        # 6. scaling
        controlnet_block_res_samples = [sample * conditioning_scale for sample in controlnet_block_res_samples]

        # import pdb
        # pdb.set_trace()
        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_res_samples,)

        return SD3ControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)


def new_forward_SD3(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        # import pdb
        # pdb.set_trace()

        Is_DreamRenderer = False
        if isinstance(hidden_states, list):
            height, width = hidden_states[0].shape[-2:]  # torch.Size([1, 16, 128, 128])
            Is_DreamRenderer = True
            feature_list = []
            bridge_image_token_index_list = []
            begin_index = 0
            for latents_index, feature in enumerate(hidden_states):
                feature = self.pos_embed(feature)
                # image_latent_len = feature.shape[1]
                feature_list.append(feature)
                bridge_image_token_index_list.append(list(range(begin_index, begin_index + feature.shape[1])))
                begin_index = begin_index + feature.shape[1]
            hidden_states = torch.cat(feature_list, dim=1)
            joint_attention_kwargs['bridge_image_token_index_list'] = bridge_image_token_index_list
        else:   
            height, width = hidden_states.shape[-2:]  # torch.Size([1, 16, 128, 128])

            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too. torch.Size([1, 4096, 1536])
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

            joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    joint_attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if Is_DreamRenderer:
            hidden_states = hidden_states[:, :len(bridge_image_token_index_list[0]), :]
        
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)