import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention

is_npu = False
try:
    import torch_npu
    is_npu = True
except:
    pass

class FluxRDIMGAttnProcessor2_0_NPU:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    counter = 0
    hard_bind_mask = None
    soft_bind_mask = None
    hard_bind_mask2 = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0_NPU requires PyTorch 2.0 and torch NPU, to use it, please upgrade PyTorch to 2.0 and install torch NPU"
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,  # (B, 512+((H//16)*(W//16))), 3072)
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_embed=None,
        prepare_img_ids_method=None,
        instance_text_index_lst=None,
        seq_len=None,
        instance_box_list=None,
        bridge_image_token_index_list=None,
        image_token_H_list=None,
        image_token_W_list=None,
        position_mask_list=None,
        global_prompt_limit_steps = 5,
        hard_image_attribute_binding_list = [],
        now_steps=None,
        hard_control_steps=20,
        num_global_text_token=200,
        num_inference_steps=20
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)  # (B, 512+((H//16)*(W//16))), 3072)
        key = attn.to_k(hidden_states)  # (B, 512+((H//16)*(W//16))), 3072)
        value = attn.to_v(hidden_states)  # (B, 512+((H//16)*(W//16))), 3072)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)


        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        HW = query.shape[2] - seq_len

        atten_mask = torch.zeros(seq_len+HW, seq_len+HW, device=query.device)
        instance_num = len(instance_box_list)

        global_seq_len = num_global_text_token

        image_token_H = image_token_H_list[0] // 16
        image_token_W = image_token_W_list[0] // 16


        if instance_num == 0:
            instance_num = -1
        
        if FluxRDIMGAttnProcessor2_0_NPU.soft_bind_mask is None:
            atten_mask = torch.zeros(seq_len+HW, seq_len+HW, device=query.device)
            for i in range(instance_num + 1):
                instance_text_idxs = instance_text_index_lst[i]
                bridge_image_token_index = bridge_image_token_index_list[i]
                bridge_image_token_index = torch.tensor(bridge_image_token_index)

                atten_mask[seq_len + bridge_image_token_index, : seq_len] = 1
                atten_mask[seq_len + bridge_image_token_index, seq_len:] = 1
                atten_mask[instance_text_idxs[:, None], seq_len:] = 1
                atten_mask[instance_text_idxs[:, None], :seq_len] = 1
            
            for i in range(instance_num + 1):
                instance_text_idxs = instance_text_index_lst[i]
                bridge_image_token_index = bridge_image_token_index_list[i]
                bridge_image_token_index = torch.tensor(bridge_image_token_index)

                atten_mask[(seq_len + bridge_image_token_index)[:, None], instance_text_idxs] = 0
                atten_mask[(seq_len + bridge_image_token_index)[:, None], seq_len + bridge_image_token_index] = 0
                atten_mask[instance_text_idxs[:, None], seq_len + bridge_image_token_index] = 0
                atten_mask[instance_text_idxs[:, None], instance_text_idxs] = 0

            for i in range(instance_num):
                instance_text_idxs = instance_text_index_lst[i + 1]
                instance_img_in_patch_idxs = position_mask_list[i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
                atten_mask[seq_len + instance_img_in_patch_idxs, global_seq_len: seq_len] = 1

            for i in range(instance_num):
                instance_text_idxs = instance_text_index_lst[i + 1]
                instance_img_in_patch_idxs = position_mask_list[i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
                atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], instance_text_idxs] = 0
                atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], seq_len + instance_img_in_patch_idxs] = 0
            atten_mask = atten_mask.bool()
            FluxRDIMGAttnProcessor2_0_NPU.soft_bind_mask = atten_mask

    


        if FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask is None:
            atten_mask = torch.zeros(seq_len+HW, seq_len+HW, device=query.device)
            for i in range(instance_num + 1):
                instance_text_idxs = instance_text_index_lst[i]
                bridge_image_token_index = bridge_image_token_index_list[i]
                bridge_image_token_index = torch.tensor(bridge_image_token_index)

                atten_mask[seq_len + bridge_image_token_index, : seq_len] = 1
                atten_mask[seq_len + bridge_image_token_index, seq_len:] = 1
                atten_mask[instance_text_idxs[:, None], seq_len:] = 1
                atten_mask[instance_text_idxs[:, None], :seq_len] = 1
            
            for i in range(instance_num + 1):
                instance_text_idxs = instance_text_index_lst[i]
                bridge_image_token_index = bridge_image_token_index_list[i]
                bridge_image_token_index = torch.tensor(bridge_image_token_index)

                atten_mask[(seq_len + bridge_image_token_index)[:, None], instance_text_idxs] = 0
                atten_mask[(seq_len + bridge_image_token_index)[:, None], seq_len + bridge_image_token_index] = 0
                atten_mask[instance_text_idxs[:, None], seq_len + bridge_image_token_index] = 0
                atten_mask[instance_text_idxs[:, None], instance_text_idxs] = 0

            for i in range(instance_num):
                instance_text_idxs = instance_text_index_lst[i + 1]
                instance_img_in_patch_idxs = position_mask_list[i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
                atten_mask[seq_len + instance_img_in_patch_idxs, : seq_len] = 1
                atten_mask[seq_len + instance_img_in_patch_idxs, seq_len:] = 1

            for i in range(instance_num):
                instance_text_idxs = instance_text_index_lst[i + 1]
                instance_img_in_patch_idxs = position_mask_list[i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
                atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], instance_text_idxs] = 0
                atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], seq_len + instance_img_in_patch_idxs] = 0
            atten_mask = atten_mask.bool()
            FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask = atten_mask


        if FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask2 is None:
            atten_mask = torch.zeros(seq_len+HW, seq_len+HW, device=query.device)
            for i in range(instance_num + 1):
                instance_text_idxs = instance_text_index_lst[i]
                bridge_image_token_index = bridge_image_token_index_list[i]
                bridge_image_token_index = torch.tensor(bridge_image_token_index)

                atten_mask[seq_len + bridge_image_token_index, : seq_len] = 1
                atten_mask[seq_len + bridge_image_token_index, seq_len:] = 1
                atten_mask[instance_text_idxs[:, None], seq_len:] = 1
                atten_mask[instance_text_idxs[:, None], :seq_len] = 1
            
            for i in range(instance_num + 1):
                instance_text_idxs = instance_text_index_lst[i]
                bridge_image_token_index = bridge_image_token_index_list[i]
                bridge_image_token_index = torch.tensor(bridge_image_token_index)

                atten_mask[(seq_len + bridge_image_token_index)[:, None], instance_text_idxs] = 0
                atten_mask[(seq_len + bridge_image_token_index)[:, None], seq_len + bridge_image_token_index] = 0
                atten_mask[instance_text_idxs[:, None], seq_len + bridge_image_token_index] = 0
                atten_mask[instance_text_idxs[:, None], instance_text_idxs] = 0

            for i in range(instance_num):
                instance_text_idxs = instance_text_index_lst[i + 1]
                instance_img_in_patch_idxs = position_mask_list[i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
                atten_mask[seq_len + instance_img_in_patch_idxs, : seq_len] = 1

            for i in range(instance_num):
                instance_text_idxs = instance_text_index_lst[i + 1]
                instance_img_in_patch_idxs = position_mask_list[i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0].to(query.device)
                atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], instance_text_idxs] = 0
            atten_mask = atten_mask.bool()
            FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask2 = atten_mask
        
        if FluxRDIMGAttnProcessor2_0_NPU.counter % 57 in hard_image_attribute_binding_list:
            if now_steps < hard_control_steps:
                atten_mask = FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask
            else:
                atten_mask = FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask2
        else:
            atten_mask = FluxRDIMGAttnProcessor2_0_NPU.soft_bind_mask
        

        


        FluxRDIMGAttnProcessor2_0_NPU.counter += 1

        

        if is_npu:
            hidden_states = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                attn.heads,
                atten_mask=atten_mask,
                input_layout="BNSD",
                pse=None,
                scale=1.0 / math.sqrt(query.shape[-1]),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=torch.logical_not(atten_mask))
        
        if FluxRDIMGAttnProcessor2_0_NPU.counter % (num_inference_steps * (19 + 38)) == 0:
            print('clear attn mask')
            FluxRDIMGAttnProcessor2_0_NPU.counter = 0
            FluxRDIMGAttnProcessor2_0_NPU.soft_bind_mask = None
            FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask = None 
            FluxRDIMGAttnProcessor2_0_NPU.hard_bind_mask2 = None 
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states