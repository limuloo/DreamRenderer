import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from PIL import Image
from huggingface_hub import login
import numpy as np

from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image

from diffusers.utils.import_utils import is_torch_npu_available
# from migc.migc_utils import get_all_processor_keys
import argparse
import os
import yaml
from tqdm import tqdm
# from migc.migc_utils import seed_everything

import sys
sys.path.insert(0, "./segment-anything-2")



parser = argparse.ArgumentParser(description='DreamRenderer SD3 Inference')
# ../output/exp011/del_color_0_
# /home/zdw/project/output/exp011_131

# get this file's name
file_name = os.path.basename(__file__)
exp_name = file_name.split('_')[0]
parser.add_argument('--num_hard_control_steps', type=int, default=20)
parser.add_argument('--use_sam_enhance', action='store_true')
parser.add_argument('--hard_image_attribute_binding_l', default=10, type=int)
parser.add_argument('--hard_image_attribute_binding_r', default=17, type=int)
parser.add_argument('--CN_hard_image_attribute_binding_l', default=0, type=int)
parser.add_argument('--CN_hard_image_attribute_binding_r', default=0, type=int)

# default parameters of SD3 ControlNet, https://huggingface.co/InstantX/SD3-Controlnet-Depth
parser.add_argument('--controlnet_conditioning_scale', default=0.5, type=float)
parser.add_argument('--control_guidance_end', default=1.0, type=float)

args = parser.parse_args()



if __name__ == '__main__':
    import torch
    from DreamRenderer.utils import get_all_processor_keys
    from DreamRenderer.processor_SD3 import JointAttnProcessor2_0
    from DreamRenderer.processor_CN_SD3 import JointAttnProcessor2_0_CN
    from DreamRenderer.pipeline_SD3 import StableDiffusion3ControlNetPipeline
    from diffusers.models import SD3ControlNetModel
    from diffusers.utils import load_image

    controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth")
    from DreamRenderer.utils import new_forward_SD3_CN
    import types
    controlnet.forward = types.MethodType(new_forward_SD3_CN, controlnet)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet=controlnet
    )
    from DreamRenderer.utils import new_forward_SD3
    pipe.transformer.forward = types.MethodType(new_forward_SD3, pipe.transformer)
    pipe.to("cuda", torch.float16)

    all_processor_keys_flux = get_all_processor_keys(pipe.transformer)
    attn_processors = {}
    for key in all_processor_keys_flux:
        attn_processors[key] = JointAttnProcessor2_0()
    pipe.transformer.set_attn_processor(attn_processors)

    # set DreamRenderer to ControlNet
    all_processor_keys_CN = get_all_processor_keys(controlnet)
    attn_processors = {}
    for key in all_processor_keys_CN:
        attn_processors[key] = JointAttnProcessor2_0_CN()
    controlnet.set_attn_processor(attn_processors)

    if args.use_sam_enhance:
        # Construct SAM Predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        sam2_checkpoint = "pretrained_weights/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        sam_predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
        sam_predictor = SAM2ImagePredictor(sam_predictor)




    seed = 42

    controlnet_conditioning_scale = args.controlnet_conditioning_scale
    control_guidance_end = args.control_guidance_end

    prompt_final = [['a photo of a black potted plant and a yellow refrigerator and a brown surfboard. a black potted plant a brown surfboard a yellow refrigerator', 
                     'a black potted plant.', 'a brown surfboard.', 'a yellow refrigerator.']]
    instance_box_list = [[0.5717187499999999, 0.0, 0.8179531250000001, 0.29807511737089204], 
                         [0.85775, 0.058755868544600943, 0.9991875, 0.646525821596244], 
                         [0.6041562500000001, 0.284906103286385, 0.799046875, 0.9898591549295774]]
            
            
    depth_path = os.path.join("./data/room_depth.png")
    control_image = Image.open(depth_path)
    control_image = control_image.convert("RGB")
    control_image = np.array(control_image)
    control_image = Image.fromarray(control_image)


    n_prompt = "bad hands, blurry, NSFW, nude, naked, porn, ugly, bad quality, worst quality"
    image = pipe(
        prompt=prompt_final[0][0],
        negative_prompt=n_prompt, 
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_end=control_guidance_end,
        guidance_scale=7.0,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    image.save(f"./sd3_{seed}.png")


    image = pipe(
        prompt='$BREAKFLAG$'.join(prompt_final[0]),
        negative_prompt=n_prompt, 
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=7.0,
        generator=torch.Generator().manual_seed(seed),instance_box_list = instance_box_list,
        hard_control_steps = args.num_hard_control_steps,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None,
        hard_image_attribute_binding_list = list(range(args.hard_image_attribute_binding_l, args.hard_image_attribute_binding_r)),
        CN_hard_image_attribute_binding_list = list(range(args.CN_hard_image_attribute_binding_l, args.CN_hard_image_attribute_binding_r)),
        control_guidance_end=control_guidance_end,
    ).images[0]
    image.save(f"./sd3_ours_{seed}.png")