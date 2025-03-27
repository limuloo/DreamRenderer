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

    if not os.path.exists("/mnt/sda/zdw/ckpt/flux_canny"):
        flux_path = "black-forest-labs/FLUX.1-Canny-dev"
    else:
        flux_path = "/mnt/sda/zdw/ckpt/flux_canny"

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


    seed = 610926

    prompt1 = 'A clock using birds stickers in place of numbers.'
    prompts = ['A clock using birds stickers in place of numbers.', 
               'a red duck sticker.',
               'a green parrot sticker.',
               'a golden eagle sticker. The eagle\'s plumage is rendered in rich golden-brown tones, with subtle highlights of amber and deep bronze to give it a metallic, almost shimmering effect. Its sharp, piercing eyes are a bold yellow, exuding intensity and focus, while the beak is curved and razor-sharp, colored in a darker gold or matte silver for contrast.',
               'a brown sparrow sticker.',
               'a purple bird sticker.',
               'a black crow sticker.',
               'a white dove sticker.']
    instance_box_list = [[0.6571, 0.25725333333333333, 0.86234, 0.40781333333333336],
                         [0.16094, 0.25536000000000003, 0.34764, 0.4764],
                         [0.39377999999999996, 0.21349333333333334, 0.6437999999999999, 0.39698666666666665],
                         [0.82584, 0.7797866666666667, 1.0, 0.9370666666666667],
                         [0.0019399999999999999, 0.48312, 0.16948, 0.71688],
                         [0.00254, 0.7686666666666667, 0.20945999999999998, 0.9777333333333332],
                         [0.8039, 0.49829333333333337, 0.9839199999999999, 0.6866933333333333]]
    control_image = Image.open('data/birds_canny.png').convert('RGB')


    images = pipe(
        prompt=prompt1,
        prompt_2=' '.join(prompts),
        control_image=control_image,
        height=args.res,
        width=args.res,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=30.0,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = [],
        hard_control_steps = 0,
        use_sam_enhance = False,
        sam_predictor = None
    ).images
    images[0][0].save(f'./birds_flux_{seed}.png')

    images = pipe(
        prompt=prompt1,
        prompt_2='$BREAKFLAG$'.join(prompts),
        control_image=control_image,
        height=args.res,
        width=args.res,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=30.0,
        generator=torch.Generator().manual_seed(seed),
        instance_box_list = instance_box_list,
        hard_control_steps = args.num_hard_control_steps,
        use_sam_enhance = args.use_sam_enhance,
        sam_predictor = sam_predictor if args.use_sam_enhance else None,
        hard_image_attribute_binding_list = args.hard_image_attribute_binding_list,
        num_instance_text_token = args.num_instance_text_token,
        num_global_text_token = args.num_global_text_token, 
    ).images
    images[0][0].save(f'./birds_ours_{seed}.png')