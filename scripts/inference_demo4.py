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
from DreamRenderer.utils import get_all_processor_keys
import argparse
import os
import yaml
from tqdm import tqdm
from DreamRenderer.utils import seed_everything
import numpy as np

device = "cuda"

def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask

@torch.no_grad()
def get_model_inputs(meta, model, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise=None, instance_input=False):
    if not instance_input:
        # update config from args
        config.update( vars(args) )
        config = OmegaConf.create(config)

    # prepare a batch of samples
    batch = prepare_batch(meta, batch=config.num_images, max_objs=30, model=clip_model, processor=clip_processor, image_size=model.image_size, use_masked_att=True, device="cuda")
    context = text_encoder.encode(  [meta["prompt"]]*config.num_images  )

    # unconditional input
    if not instance_input:
        uc = text_encoder.encode( config.num_images*[""] )
        if args.negative_prompt is not None:
            uc = text_encoder.encode( config.num_images*[args.negative_prompt] )
    else:
        uc = None

    # sampler
    if not instance_input:
        alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
        if config.mis > 0:
            sampler = PLMSSamplerInst(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale, mis=config.mis)
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50
    else:
        sampler, steps = None, None

    # grounding input
    grounding_input = grounding_tokenizer_input.prepare(batch, return_att_masks=return_att_masks)

    # model inputs
    input = dict(x = starting_noise, timesteps = None, context = context, grounding_input = grounding_input)
    return input, sampler, steps, uc, config

@torch.no_grad()
def run(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise=None, guidance_scale=None):
    # prepare models inputs
    input, sampler, steps, uc, config = get_model_inputs(meta, model, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, instance_input=False)
    if guidance_scale is not None:
        config.guidance_scale = guidance_scale
    
    # prepare models inputs for each instance if MIS is used
    if args.mis > 0:
        input_all = [input]
        for i in range(len(meta['phrases'])):
            meta_instance = prepare_instance_meta(meta, i)
            input_instance, _, _, _, _ = get_model_inputs(meta_instance, model, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, instance_input=True)
            input_all.append(input_instance)
    else:
        input_all = input

    # start sampling
    shape = (config.num_images, model.in_channels, model.image_size, model.image_size)
    with torch.autocast(device_type=device, dtype=torch.float16):
        samples_fake = sampler.sample(S=steps, shape=shape, input=input_all,  uc=uc, guidance_scale=config.guidance_scale)
    samples_fake = autoencoder.decode(samples_fake)

    # define output folder
    output_folder = os.path.join( args.output,  meta["save_folder_name"])
    os.makedirs( output_folder, exist_ok=True)

    start = len( os.listdir(output_folder) )
    image_ids = list(range(start,start+config.num_images))
    # print(image_ids)
    
    # visualize the boudning boxes
    image_boxes = draw_boxes( meta["locations"], meta["phrases"], meta["prompt"] + ";alpha=" + str(meta['alpha_type'][0]) )
    img_name = os.path.join( output_folder, str(image_ids[0])+'_boxes.png' )
    image_boxes.save( img_name )
    print("saved image with boxes at {}".format(img_name))
    
    # if use cascade model, we will use SDXL-Refiner to refine the generated images
    if config.cascade_strength > 0:
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        pipe = pipe.to("cuda:0")
        strength, steps = config.cascade_strength, 20 # default setting, need to be manually tuned.

    # save the generated images
    for image_id, sample in zip(image_ids, samples_fake):
        img_name = str(int(image_id))+'.png'
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.cpu().numpy().transpose(1,2,0) * 255 
        sample = Image.fromarray(sample.astype(np.uint8))
        if config.cascade_strength > 0:
            prompt = meta["prompt"]
            refined_image = pipe(prompt, image=sample, strength=strength, num_inference_steps=steps).images[0]
            refined_image.save(  os.path.join(output_folder, img_name.replace('.png', '_xl_s{}_n{}.png'.format(strength, steps)))   )
        sample.save(  os.path.join(output_folder, 'instance_diffusion_output.png')   )

def rescale_box(bbox, width, height):
    x0 = bbox[0]/width
    y0 = bbox[1]/height
    x1 = (bbox[0]+bbox[2])/width
    y1 = (bbox[1]+bbox[3])/height
    return [x0, y0, x1, y1]

def get_point_from_box(bbox):
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return [(x0 + x1)/2.0, (y0 + y1)/2.0]

def rescale_points(point, width, height):
    return [point[0]/float(width), point[1]/float(height)]

def rescale_scribbles(scribbles, width, height):
    return [[scribble[0]/float(width), scribble[1]/float(height)] for scribble in scribbles]
    
# draw boxes given a lits of boxes: [[top left cornor, top right cornor, width, height],]
# show descriptions per box if descriptions is not None
def draw_boxes(boxes, descriptions=None, caption=None):
    width, height = 512, 512
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)   
    boxes = [ [ int(x*width) for x in box ] for box in boxes]
    for i, box in enumerate(boxes):
        draw.rectangle( ( (box[0], box[1]), (box[2], box[3]) ), outline=(0,0,0), width=2)
    if descriptions is not None:
        for idx, box in enumerate(boxes):
            draw.text((box[0], box[1]), descriptions[idx], fill="black")
    if caption is not None:
        draw.text((0, 0), caption, fill=(255,102,102))
    return image

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,  default="./", help="root folder for output")
    parser.add_argument("--num_images", type=int, default=1, help="")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--negative_prompt", type=str,  default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="")
    # parser.add_argument("--input_json", type=str, default='demos/demo_cat_dog_robin.json', help="json files for instance-level conditions")
    # parser.add_argument("--ckpt", type=str, default='pretrained/instancediffusion_sd15.pth', help="pretrained checkpoint")
    # parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--alpha", type=float, default=0.8, help="the percentage of timesteps using grounding inputs")
    parser.add_argument("--mis", type=float, default=0.36, help="the percentage of timesteps using MIS")
    parser.add_argument("--cascade_strength", type=float, default=0.0, help="strength of SDXL Refiner.")
    parser.add_argument("--test_config", type=str, default="L2I_methods/InstanceDiffusion/configs/test_box.yaml", help="config for model inference.")
    
    parser.add_argument('--res', type=int, default=1024)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--num_hard_control_steps', type=int, default=20)
    parser.add_argument('--use_sam_enhance', action='store_true')
    parser.add_argument('--hard_image_attribute_binding_list', default=list(range(19, 38)), type=list)
    parser.add_argument('--num_instance_text_token', default=200, type=int)
    parser.add_argument('--num_global_text_token', default=200, type=int)

    args = parser.parse_args()

    return_att_masks = False
    # ckpt = args.ckpt
    ckpt = "pretrained_weights/instancediffusion_sd15.pth"
    
    seed = 66666
    save_folder_name = f"./"

    # read json files
    # with open(args.input_json) as f:
    #     data = json.load(f)

    '''
    {"caption": "a American robin, brown Maltipoo dog, a gray British Shorthair in a stream, alongside with trees and rocks", 
"width": 512, "height": 512, 
"annos": [
{"bbox": [0, 51, 179, 230], "mask": [], "category_name": "", "caption": "a gray British Shorthair standing on a rock in the woods"}, 
{"bbox": [179, 102, 153, 153], "mask": [], "category_name": "", "caption": "a yellow American robin standing on the rock"},
{"bbox": [332, 102, 179, 255], "mask": [], "category_name": "", "caption": "a brown Maltipoo dog standing on the rock"},
{"bbox": [0, 358, 512, 153], "mask": [], "category_name": "", "caption": "a close up of a small waterfall in the woods"}]}
    '''
    prompt_final = [["a bird made by fire, a lego dog, a cat made by ice, alongside with trees and rocks",
                     'a bird made by fire standing on a rock in the woods', 
                     'a lego dog standing on the rock', 
                     'a cat made by ice standing on the rock', 
                     'a close up of a small waterfall in the woods']]
    input_bboxes = [[[0.0, 0.099609375, 0.349609375, 0.548828125], 
                     [0.349609375, 0.19921875, 0.6484375, 0.498046875], 
                     [0.6484375, 0.19921875, 0.998046875, 0.697265625], 
                     [0.0, 0.69921875, 1.0, 0.998046875]]]

    ########################################################## Begin Instance-Diffusion Inference ##########################################################
    bboxes = [[]]
    for o in input_bboxes[0]:
        x0, y0, x1, y1 = o[0], o[1], o[2], o[3]
        bboxes[0].append([x0, y0, (x1 - x0), (y1 - y0)])
    bboxes = [[[int(x*512) for x in box] for box in bboxes[0]]]
    data = {}
    data["caption"] = prompt_final[0][0]
    data["annos"] = []
    for i in range(len(bboxes[0])):
        data["annos"].append({"bbox": bboxes[0][i], "mask": [], "category_name": "", "caption": prompt_final[0][i + 1]})
    data["width"] = 512
    data["height"] = 512
    # import pdb; pdb.set_trace()

    # START: READ BOXES AND BINARY MASKS
    boxes = []
    binay_masks = []
    # class_names = []
    instance_captions = []
    points_list = []
    scribbles_list = []
    prompt = data['caption']
    crop_mask_image = False
    for inst_idx in range(len(data['annos'])):
        if "mask" not in data['annos'][inst_idx] or data['annos'][inst_idx]['mask'] == []:
            instance_mask = np.zeros((512,512,1))
        else:
            instance_mask = decodeToBinaryMask(data['annos'][inst_idx]['mask'])
            if crop_mask_image:
                # crop the instance_mask to 512x512, centered at the center of the instance_mask image
                # get the center of the instance_mask
                center = np.array([instance_mask.shape[0]//2, instance_mask.shape[1]//2])
                # get the top left corner of the crop
                top_left = center - np.array([256, 256])
                # get the bottom right corner of the crop
                bottom_right = center + np.array([256, 256])
                # crop the instance_mask
                instance_mask = instance_mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                binay_masks.append(instance_mask)
                data['width'] = 512
                data['height'] = 512
            else:
                binay_masks.append(instance_mask)

        if "bbox" not in data['annos'][inst_idx]: 
            boxes.append([0,0,0,0])
        else:
            boxes.append(data['annos'][inst_idx]['bbox'])
        if 'point' in data['annos'][inst_idx]:
            points_list.append(data['annos'][inst_idx]['point'])
        if "scribble" in data['annos'][inst_idx]:
            scribbles_list.append(data['annos'][inst_idx]['scribble'])
        # class_names.append(data['annos'][inst_idx]['category_name'])
        instance_captions.append(data['annos'][inst_idx]['caption'])
        # show_binary_mask(binay_masks[inst_idx])
    # import pdb; pdb.set_trace()

    # END: READ BOXES AND BINARY MASKS
    img_info = {}
    img_info['width'] = data['width']
    img_info['height'] = data['height']

    locations = [rescale_box(box, img_info['width'], img_info['height']) for box in boxes]
    phrases = instance_captions

    # get points for each instance, if not provided, use the center of the box
    if len(points_list) == 0:
        points = [get_point_from_box(box) for box in locations] 
    else: 
        points = [rescale_points(point, img_info['width'], img_info['height']) for point in points_list] 

    # get binary masks for each instance, if not provided, use all zeros
    binay_masks = []
    for i in range(len(locations) - len(binay_masks)):
        binay_masks.append(np.zeros((512,512,1)))

    # get scribbles for each instance, if not provided, use random scribbles
    if len(scribbles_list) == 0:
        for idx, mask_fg in enumerate(binay_masks):
            # get scribbles from segmentation if scribble is not provided
            scribbles = sample_random_points_from_mask(mask_fg, 20)
            scribbles = convert_points(scribbles, img_info)
            scribbles_list.append(scribbles)
    else:
        scribbles_list = [rescale_scribbles(scribbles, img_info['width'], img_info['height']) for scribbles in scribbles_list]
        scribbles_list = reorder_scribbles(scribbles_list)

    print("num of inst captions, masks, boxes and points: ", len(phrases), len(binay_masks), len(locations), len(points))

    # get polygons for each annotation
    polygons_list = []
    segs_list = []
    for idx, mask_fg in enumerate(binay_masks):
        # binary_mask = mask_fg[:,:,0]
        polygons = sample_sparse_points_from_mask(mask_fg, k=256)
        if polygons is None:
            polygons = [0 for _ in range(256*2)]
        polygons = convert_points(polygons, img_info)
        polygons_list.append(polygons)

        segs_list.append(resize(mask_fg.astype(np.float32), (512, 512, 1)))

    segs = np.stack(segs_list).astype(np.float32).squeeze() if len(segs_list) > 0 else segs_list
    polygons = polygons_list
    scribbles = scribbles_list

    meta_list = [ 
        # grounding inputs for generation
        dict(
            ckpt = ckpt,
            prompt = prompt,
            phrases = phrases,
            polygons = polygons,
            scribbles = scribbles,
            segs = segs,
            locations = locations,
            points = points, 
            alpha_type = [args.alpha, 0.0, 1-args.alpha],
            save_folder_name=save_folder_name
        ), 
    ]

    # set seed
    torch.manual_seed(seed)
    starting_noise = torch.randn(args.num_images, 4, 64, 64).to(device)

    model, autoencoder, text_encoder, diffusion, config = load_model_ckpt(meta_list[0]["ckpt"], args, device)
    clip_model, clip_processor = create_clip_pretrain_model()

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    for meta in meta_list:
        run(meta, model, autoencoder, text_encoder, diffusion, clip_model, clip_processor, config, grounding_tokenizer_input, starting_noise, guidance_scale=args.guidance_scale)

    ########################################################## End Instance-Diffusion Inference ##########################################################



    ########################################################## Begin Re-Render ##########################################################
    bboxes = input_bboxes
    def construct_depth_anything_v2_model():
        model = DepthAnythingV2(**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]})
        model.load_state_dict(torch.load('pretrained_weights/depth_anything_v2_vitl.pth', map_location='cpu'))
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = model.to(DEVICE).eval()
        return model
    model = construct_depth_anything_v2_model()

    import cv2
    raw_img = cv2.imread('instance_diffusion_output.png')
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    cv2.imwrite('depth.png', depth)



    flux_path = "black-forest-labs/FLUX.1-Depth-dev"

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

    # import pdb; pdb.set_trace()


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
        sam_image = Image.open('instance_diffusion_output.png').convert('RGB')
    ).images
    images[0][0].save('./instance_diffusion_output_dreamrenderer.png')

    ########################################################## End Re-Render ##########################################################