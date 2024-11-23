import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from distributed import init_distributed_device
import argparse
import json
import datetime
import sys
import logging
import os
import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--input-json", type=str, default="",help="json file path")
    parser.add_argument("--batch-size",type=int,default=3)
    parser.add_argument("--numgpus", type=int, default=1)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument('--world-size', type=int, default=1, help='number of nodes for distributed training')
    parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank on this node')
    parser.add_argument('--horovod', default=False, action='store_true', help='Use Horovod for distributed training')
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )    
    args = parser.parse_args()
    return args


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values.to(torch.bfloat16) #.to(torch.half)

if __name__ == "__main__":
    print('Initializing Chat')
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('aaa.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    rank = int(os.environ['RANK'])
    logging.info(f"rank: {rank}")
    args.horovod = False
    device = init_distributed_device(args)
    logging.info(f"device: {device}")
    path = 'OpenGVLab/InternVL2-8B' # set the path to the model
    model = AutoModel.from_pretrained(
    path,
    device_map = {"": device},
    # load_in_8bit=True,
    torch_dtype=torch.bfloat16,    
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval()

    logging.info(f"model: {model.device}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    generation_config = dict(
        num_beams=1,
        max_new_tokens=77,
        do_sample=False,
    )
    print('Initialization Finished')
    if args.input_json == "" or not os.path.exists(args.input_json):
        input_json = 'json file path for cc12m dataset'
    else:
        input_json = args.input_json
    print(f'load from {input_json}')
    out_json = input_json[:-5]+f"_internvl_cc12m_{rank}"+".json"
    data_list = []
    image_list = []
    image_pack_list = []
    number=0
    batch_size = args.batch_size
    with open(input_json, 'r') as json_file:
        data = json.load(json_file)
    data_length = len(data)
    split_length = data_length//args.numgpus
    start = rank*split_length
    end = (rank+1)*split_length if rank!=args.numgpus-1 else data_length
    for i in range(start,end):
        image_list.append(data[i]['image'])
        number+=1
        if number==batch_size:
            image_pack_list.append(image_list)
            image_list = []
            number=0
        if i == (end-1) and len(image_list)!=0:
            image_pack_list.append(image_list)

    if os.path.exists(out_json): #resume from the last image
        continue_point = 0
        with open(out_json, 'r') as file:
            json_text = file.read()
        substring = "image folder prefix"
        last_position = json_text.rfind(substring)
        substring_after_last = json_text[last_position:]
        first_jpg_position = substring_after_last.find("jpg")
        last_image_path = json_text[last_position:last_position+first_jpg_position+3]
        for i in range(len(image_pack_list)):
            flag=0
            image_list = image_pack_list[i]
            for j in range(len(image_list)):
                if image_list[j] == last_image_path:
                    continue_point = i
                    flag=1
                    break
            if flag==1:
                break
        index = json_text.find(image_pack_list[continue_point][0])
        json_text = json_text[:index-11]
        with open(out_json, 'w') as file:
            file.write(json_text)

    prompt = '<image>\nDescribe the image in detail.'
    # prompt = "<image>\nBriefly describe the image with just a few words."
    # prompt = "<image>\nBriefly describe the image in one sentence."
    # prompt = "<image>\nBriefly describe the image with few noun words separated by ','."
    # prompt = "<image>\nDescribe the style or your feelings about this image, do not say anything about the objects in the image."
    # prompt = "<image>\nDescribe only the background of this image, do not say anything about the foreground objects."
    # prompt = "<image>\nDescribe only the one main object in the image, do not say anything about the other objects or background."

    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for i in tqdm(range(continue_point, len(image_pack_list))):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]

                image_list = [load_image(image, max_num=6) for image in image_url_list]
                num_patches_list = [image.size(0) for image in image_list]
                pixel_values = torch.cat(image_list, dim=0).to(device)
                question_list = [prompt]*len(num_patches_list)
                tokenizer.padding_side = 'left'
                queries,eos_token_id,template_sep = model.question2query(question_list,num_patches_list=num_patches_list, tokenizer=tokenizer)
                model_inputs = tokenizer(queries, return_tensors='pt', padding=True)         
                model_inputs['input_ids'] = model_inputs['input_ids'].to(device)
                model_inputs['attention_mask'] = model_inputs['attention_mask'].to(device)       
                ii = 10
                for _ in range(ii):
                    try:
                        responses = model.batch_chat(tokenizer, pixel_values,
                                                    num_patches_list=num_patches_list,
                                                    questions=question_list,
                                                    generation_config=generation_config,
                                                    device=device, model_inputs=model_inputs,
                                                    eos_token_id=eos_token_id,
                                                    template_sep=template_sep)
                        break
                    except:
                        if _ == ii-1:
                            responses = model.batch_chat(tokenizer, pixel_values,
                                                        num_patches_list=num_patches_list,
                                                        questions=question_list,
                                                        generation_config=generation_config,
                                                        device=device, model_inputs=model_inputs,
                                                        eos_token_id=eos_token_id,
                                                        template_sep=template_sep)                
                        continue
                for j in range(len(image_url_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': responses[j]}
                    json.dump(data, outfile)
                    
                    if i==(len(image_pack_list)-1) and j==(len(image_url_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')

                print(f'{i},{len(image_pack_list)}')
                current_time = datetime.datetime.now()
                print(current_time)
                outfile.flush()
    else:
        with open(out_json, 'w') as outfile:
            for i in tqdm(range(len(image_pack_list))):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]
                
                image_list = [load_image(image, max_num=6) for image in image_url_list]
                num_patches_list = [image.size(0) for image in image_list]
                pixel_values = torch.cat(image_list, dim=0).to(device)
                question_list = [prompt]*len(num_patches_list)
                tokenizer.padding_side = 'left'
                queries,eos_token_id,template_sep = model.question2query(question_list,num_patches_list=num_patches_list, tokenizer=tokenizer)
                model_inputs = tokenizer(queries, return_tensors='pt', padding=True)         
                model_inputs['input_ids'] = model_inputs['input_ids'].to(device)
                model_inputs['attention_mask'] = model_inputs['attention_mask'].to(device)
                ii = 10
                for _ in range(ii):
                    try:
                        responses = model.batch_chat(tokenizer, pixel_values,
                                                    num_patches_list=num_patches_list,
                                                    questions=question_list,
                                                    generation_config=generation_config,
                                                    device=device, model_inputs=model_inputs,
                                                    eos_token_id=eos_token_id,template_sep=template_sep)
                        break
                    except:
                        if _ == ii-1:
                            responses = model.batch_chat(tokenizer, pixel_values,
                                                        num_patches_list=num_patches_list,
                                                        questions=question_list,
                                                        generation_config=generation_config,
                                                        device=device, model_inputs=model_inputs,
                                                        eos_token_id=eos_token_id,
                                                        template_sep=template_sep)                
                        continue
                for j in range(len(image_url_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': responses[j]}
                    json.dump(data, outfile)
                    
                    if i==(len(image_pack_list)-1) and j==(len(image_url_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')

                print(f'{i},{len(image_pack_list)}')
                current_time = datetime.datetime.now()
                print(current_time)
                outfile.flush()
