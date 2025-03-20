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
from DreamRenderer.utils import get_all_processor_keys
import argparse
import os
import yaml
from tqdm import tqdm
from DreamRenderer.utils import seed_everything
from diffusers import FluxPriorReduxPipeline, FluxPipeline



parser = argparse.ArgumentParser(description='DreamRenderer Inference')


parser.add_argument('--res', type=int, default=1024)
parser.add_argument('--num_inference_steps', type=int, default=20)
parser.add_argument('--num_hard_control_steps', type=int, default=20)
parser.add_argument('--use_sam_enhance', action='store_true')
parser.add_argument('--hard_image_attribute_binding_list', default=list(range(19, 38)), type=list)
parser.add_argument('--num_instance_text_token', default=200, type=int)
parser.add_argument('--num_global_text_token', default=200, type=int)

args = parser.parse_args()



if __name__ == '__main__':

    if not os.path.exists("/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev"):
        flux_path = "black-forest-labs/FLUX.1-Depth-dev"
    else:
        flux_path = "/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev"

    pipe = FluxControlPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to("cuda")



    if not os.path.exists("/mnt/sda/zdw/ly/ckpt/FLUX.1-Redux-dev"):
        redux_path = "black-forest-labs/FLUX.1-Redux-dev"
    else:
        redux_path = "/mnt/sda/zdw/ly/ckpt/FLUX.1-Redux-dev"
    pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(redux_path, torch_dtype=torch.bfloat16).to("cuda")



    all_processor_keys_flux = get_all_processor_keys(pipe.transformer)
    attn_processors = {}
    for key in all_processor_keys_flux:
        attn_processors[key] = FluxRDIMGAttnProcessor2_0_NPU()
    pipe.transformer.set_attn_processor(attn_processors)

    if args.use_sam_enhance:
        # Construct SAM Predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        sam2_checkpoint = "pretrained_weights/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        sam_predictor = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
        sam_predictor = SAM2ImagePredictor(sam_predictor)


    import random
    seed = 131195091

    prompt1 = 'a dog and a cat on the grass'
    prompts = ['a dog and a cat on the grass', 'data/cat.png%IMAGEFLAG%', 'data/dog.png%IMAGEFLAG%']
    instance_box_list = [[0.328125, 0, 0.703125, 0.34375], [0.109375, 0.28125, 0.859375, 1]]
    control_image = Image.open('data/cat_dog_depth.png').convert('RGB')

    images = pipe(
        prompt=prompt1,
        prompt_2='$BREAKFLAG$'.join(prompts),
        control_image=control_image,
        height=args.res,
        width=args.res,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = instance_box_list,
        hard_control_steps = args.num_hard_control_steps,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None,
        hard_image_attribute_binding_list = args.hard_image_attribute_binding_list,
        num_instance_text_token = args.num_instance_text_token,
        num_global_text_token = args.num_global_text_token, 
        Redux = pipe_prior_redux
    ).images
    images[0][0].save(f'./cat_and_dog.png')