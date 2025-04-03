import sys
sys.path.insert(0, "./L2I_methods/InstanceDiffusion")

import os
import json
import torch 
import argparse
import numpy as np

from functools import partial
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from tkinter.messagebox import NO
from diffusers.utils import load_image
from diffusers import StableDiffusionXLImg2ImgPipeline

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.plms_instance import PLMSSamplerInst
from dataset.decode_item import sample_random_points_from_mask, sample_sparse_points_from_mask, decodeToBinaryMask, reorder_scribbles

from skimage.transform import resize
from utils.checkpoint import load_model_ckpt
from utils.input import convert_points, prepare_batch, prepare_instance_meta
from utils.model import create_clip_pretrain_model, set_alpha_scale, alpha_generator
import sys
sys.path.insert(0, "./segment-anything-2")


from depth_anything_v2.dpt import DepthAnythingV2
# ASCEND_RT_VISIBLE_DEVICES=5 python teaser/pool_ball/scripts.py  --use_sam_enhance
import torch
is_npu = False
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    is_npu = True
except:
    pass
from DreamRenderer.pipeline import FluxControlPipeline
from PIL import Image
from DreamRenderer.processor import FluxRDIMGAttnProcessor2_0_NPU
from DreamRenderer.processor_CN_Union import FluxRDIMGAttnProcessor2_0_CN_Union
from DreamRenderer.utils import get_all_processor_keys
import argparse
import os
import yaml
from tqdm import tqdm
from DreamRenderer.utils import seed_everything
import numpy as np


if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--res', type=int, default=1024)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--num_hard_control_steps', type=int, default=30)
    parser.add_argument('--use_sam_enhance', action='store_true')
    parser.add_argument('--hard_image_attribute_binding_list', default=list(range(19, 38)), type=list)
    parser.add_argument('--num_instance_text_token', default=200, type=int)
    parser.add_argument('--num_global_text_token', default=200, type=int)
    parser.add_argument('--control_guidance_end', type=float, default=0.5)
    parser.add_argument('--controlnet_conditioning_scale', type=float, default=0.5)
    # controlnet_conditioning_scale

    args = parser.parse_args()
    
    seed = 6674674674


    
    prompt_final = [["three dogs.",
                     'Pug dog. A small, compact dog with a wrinkled, flat face and large, round eyes. It has a short, smooth fawn coat with a curled tail resting tightly over its back. The ears are small and folded, and the dog has a sturdy, barrel-shaped body with short legs. Its expression is alert, charming, and playful.', 
                     'Golden Retriever. A medium-to-large dog with a well-proportioned, muscular body. It has a dense, flowing golden coat that can range from light to dark gold. The dog has a broad head, kind brown eyes, and medium-sized floppy ears. Its tail is thick and feathered, carried straight with a slight curve. The overall expression is friendly, intelligent, and gentle.', 
                     'Black Labrador. A medium-to-large dog with a well-proportioned, muscular body. It has a dense, flowing black coat that can range from light to dark black. The dog has a broad head, kind brown eyes, and medium-sized floppy ears. Its tail is thick and feathered, carried straight with a slight curve. The overall expression is friendly, intelligent, and gentle.']]
    input_bboxes = [[[0, 0.25, 0.328125, 0.7656255], 
                     [0.28125, 0.265625, 0.625, 0.90625],
                      [0.578125, 0.1875, 0.984375, 0.953125]
                     ]]

    ########################################################## Begin Re-Render ##########################################################
    
    
    from DreamRenderer.pipeline_CN import FluxControlNetPipeline
    from DreamRenderer.controlnet import FluxControlNetModel
    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "InstantX/FLUX.1-dev-Controlnet-Union"

    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    # set DreamRenderer to FLUX
    all_processor_keys_flux = get_all_processor_keys(pipe.transformer)
    attn_processors = {}
    for key in all_processor_keys_flux:
        attn_processors[key] = FluxRDIMGAttnProcessor2_0_NPU()
    pipe.transformer.set_attn_processor(attn_processors)

    # set DreamRenderer to ControlNet
    all_processor_keys_CN = get_all_processor_keys(controlnet)
    attn_processors = {}
    for key in all_processor_keys_CN:
        attn_processors[key] = FluxRDIMGAttnProcessor2_0_CN_Union()
    controlnet.set_attn_processor(attn_processors)

    if args.use_sam_enhance:
        # Construct SAM Predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        sam2_checkpoint = "pretrained_weights/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        sam_predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
        sam_predictor = SAM2ImagePredictor(sam_predictor)



    prompt1 = prompt_final[0][0]
    for i in range(1, len(prompt_final[0])):
        prompt_final[0][0] = prompt_final[0][0] + ', ' + prompt_final[0][i]
    prompts = prompt_final[0]
    instance_box_list = input_bboxes[0]
    control_image = Image.open('data/dogs_depth.png').convert('RGB')


    controlnet_conditioning_scale = args.controlnet_conditioning_scale
    control_guidance_end = args.control_guidance_end

    control_mode = 2 # depth mode

    images = pipe(
        prompt=prompt1,
        prompt_2=' '.join(prompts),
        control_image=control_image,
        control_mode = control_mode,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_end=control_guidance_end,
        height=args.res,
        width=args.res,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = [],
        hard_control_steps = args.num_hard_control_steps,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None,
        hard_image_attribute_binding_list = args.hard_image_attribute_binding_list,
        num_instance_text_token = args.num_instance_text_token,
        num_global_text_token = args.num_global_text_token
    ).images
    images[0].save('./dogs_CN.png')


    images = pipe(
        prompt=prompt1,
        prompt_2='$BREAKFLAG$'.join(prompts),
        control_image=control_image,
        control_mode = control_mode,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_end=control_guidance_end,
        height=args.res,
        width=args.res,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = instance_box_list,
        hard_control_steps = args.num_hard_control_steps,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None,
        hard_image_attribute_binding_list = args.hard_image_attribute_binding_list,
        num_instance_text_token = args.num_instance_text_token,
        num_global_text_token = args.num_global_text_token
    ).images
    images[0].save('./dogs_ours.png')


    ########################################################## End Re-Render ##########################################################