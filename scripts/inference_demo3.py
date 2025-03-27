import yaml
from diffusers import EulerDiscreteScheduler

import sys
sys.path.insert(0, './L2I_methods/MIGC')


from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
import os
import torch
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
from DreamRenderer.utils import get_all_processor_keys
import argparse
import os
import yaml
from tqdm import tqdm
from DreamRenderer.utils import seed_everything
import numpy as np
# import pdb
# pdb.set_trace()


parser = argparse.ArgumentParser(description='DreamRenderer Inference')

parser.add_argument('--res', type=int, default=1024)
parser.add_argument('--num_inference_steps', type=int, default=20)
parser.add_argument('--num_hard_control_steps', type=int, default=20)
parser.add_argument('--use_sam_enhance', action='store_true')
parser.add_argument('--hard_image_attribute_binding_list', default=list(range(19, 38)), type=list)
parser.add_argument('--num_instance_text_token', default=200, type=int)
parser.add_argument('--num_global_text_token', default=200, type=int)

args = parser.parse_args()


def construct_depth_anything_v2_model():
    model = DepthAnythingV2(**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]})
    model.load_state_dict(torch.load('pretrained_weights/depth_anything_v2_vitl.pth', map_location='cpu'))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(DEVICE).eval()
    return model
    


if __name__ == '__main__':
    migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"


    sd1x_path = '/mnt/sda/zdw/ckpt/new_sd14' if os.path.isdir('/mnt/sda/zdw/ckpt/new_sd14') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    prompt_final = [['masterpiece, best quality,black colored ball,gray colored cat,white colored  bed,\
                     green colored plant,red colored teddy bear,blue colored wall,brown colored vase,orange colored book,\
                     yellow colored hat', 'black colored ball', 'gray colored cat', 'white colored  bed', 'green colored plant', \
                        'red colored teddy bear', 'blue colored wall', 'brown colored vase', 'orange colored book', 'yellow colored hat']]
    bboxes = [[[0.3125, 0.609375, 0.625, 0.875], [0.5625, 0.171875, 0.984375, 0.6875], \
               [0.0, 0.265625, 0.984375, 0.984375], [0.0, 0.015625, 0.21875, 0.328125], \
                [0.171875, 0.109375, 0.546875, 0.515625], [0.234375, 0.0, 1.0, 0.3125], \
                    [0.71875, 0.625, 0.953125, 0.921875], [0.0625, 0.484375, 0.359375, 0.8125], \
                        [0.609375, 0.09375, 0.90625, 0.28125]]]
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    seed = 7351007268695528845
    seed_everything(seed)
    image = pipe(prompt_final, bboxes, num_inference_steps=50, guidance_scale=7.5, 
                    MIGCsteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
    image.save('migc_output.png')

    model = construct_depth_anything_v2_model()

    import cv2
    raw_img = cv2.imread('migc_output.png')
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    cv2.imwrite('depth.png', depth)



    if not os.path.exists("/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev"):
        flux_path = "black-forest-labs/FLUX.1-Depth-dev"
    else:
        flux_path = "/mnt/sda/zdw/ckpt/FLUX.1-Depth-dev"

    pipe = FluxControlPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16).to("cuda")
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



    prompt1 = prompt_final[0][0]
    prompts = prompt_final[0]
    instance_box_list = bboxes[0]
    control_image = Image.open('./depth.png').convert('RGB')


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
        sam_image = Image.open('migc_output.png').convert('RGB')
    ).images
    images[0][0].save('./migc_dreamrenderer.png')

