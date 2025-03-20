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


    seed = 567693402272770

    prompt1 = 'sixteen balls.'
    prompts = ['sixteen balls..', 'A shiny red pool ball resting on the table, prominently engraved with a bold white circle containing the number "1" clearly visible on its glossy surface..', "A shiny blue pool ball resting on the table, prominently engraved with a bold white circle containing the number '2' clearly visible on its glossy surface..", "A shiny yellow pool ball resting on the table, prominently engraved with a bold white circle containing the number '3' clearly visible on its glossy surface..", "A shiny green pool ball resting on the table, prominently engraved with a bold white circle containing the number '4' clearly visible on its glossy surface..", "A shiny purple pool ball resting on the table, prominently engraved with a bold white circle containing the number '5' clearly visible on its glossy surface..", "A shiny orange pool ball resting on the table, prominently engraved with a bold white circle containing the number '6' clearly visible on its glossy surface..", "A shiny magenta pool ball resting on the table, prominently engraved with a bold white circle containing the number '7' clearly visible on its glossy surface..", "A shiny black pool ball resting on the table, prominently engraved with a bold white circle containing the number '8' clearly visible on its glossy surface..", "A shiny pink pool ball resting on the table, prominently engraved with a bold white circle containing the number '9' clearly visible on its glossy surface..", "A shiny turquoise pool ball resting on the table, prominently engraved with a bold white circle containing the number '10' clearly visible on its glossy surface..", "A shiny gold pool ball resting on the table, prominently engraved with a bold white circle containing the number '11' clearly visible on its glossy surface..", "A shiny silver pool ball resting on the table, prominently engraved with a bold white circle containing the number '12' clearly visible on its glossy surface..", "A shiny maroon pool ball resting on the table, prominently engraved with a bold white circle containing the number '13' clearly visible on its glossy surface..", "A shiny teal pool ball resting on the table, prominently engraved with a bold white circle containing the number '14' clearly visible on its glossy surface..", "A shiny lavender pool ball resting on the table, prominently engraved with a bold white circle containing the number '15' clearly visible on its glossy surface..", 'A pure white pool ball..']
    instance_box_list = [[0.046875, 0.046875, 0.265625, 0.28125], [0.265625, 0.046875, 0.484375, 0.28125], [0.484375, 0.046875, 0.7109375, 0.28125], [0.7109375, 0.046875, 0.9296875, 0.28125], [0.046875, 0.28125, 0.265625, 0.484375], [0.265625, 0.28125, 0.484375, 0.484375], [0.484375, 0.28125, 0.7109375, 0.484375], [0.7109375, 0.28125, 0.9375, 0.4921875], [0.046875, 0.484375, 0.265625, 0.7109375], [0.265625, 0.484375, 0.4921875, 0.7109375], [0.4921875, 0.484375, 0.71875, 0.7109375], [0.71875, 0.4921875, 0.9453125, 0.7109375], [0.046875, 0.7109375, 0.265625, 0.9296875], [0.265625, 0.7109375, 0.4921875, 0.9453125], [0.4921875, 0.7109375, 0.71875, 0.9453125], [0.71875, 0.7109375, 0.9453125, 0.9453125]]
    control_image = Image.open('data/balls_depth.png').convert('RGB')


    images = pipe(
        prompt=prompt1,
        prompt_2=' '.join(prompts),
        control_image=control_image,
        height=args.res,
        width=args.res,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=10.0,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = [],
        hard_control_steps = 0,
        use_sam_enhance = False,
        sam_predictor = None
    ).images
    images[0][0].save('./balls_flux.png')

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
    ).images
    images[0][0].save('./balls_ours.png')