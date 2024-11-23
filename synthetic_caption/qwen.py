import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from distributed import init_distributed_device
import argparse
import json
import datetime
import sys
import logging
import os
import sys
import ipdb
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--batch-size",type=int,default=3)
    parser.add_argument("--numgpus", type=int, default=1)
    parser.add_argument("--input-json", type=str, default="",help="json file path")
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


def dialogue_format(image_path, prompt):
    return [{"role": "user","content": [{"type": "image", "image": image_path},{"type": "text", "text": prompt}]}]

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
    path = 'Qwen-VL/Qwen2-VL-7B-Instruct'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map = {"": device}, use_cache=True
    ).eval()
    # , attn_implementation="flash_attention_2", use_cache=True
    logging.info(f"model: {model.device}")

    processor = AutoProcessor.from_pretrained(path)

    print('Initialization Finished')
    if args.input_json == "" or not os.path.exists(args.input_json):
        input_json = 'json file path for cc12m dataset'
    else:
        input_json = args.input_json
    print(f'load from {input_json}')
    out_json = input_json[:-5]+f"_qwenvl_cc12m_{rank}"+".json"
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

    if os.path.exists(out_json): # if the file already exists, continue from the last image
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

    prompt = 'Describe the image in detail:'

    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for i in tqdm(range(continue_point, len(image_pack_list))):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]
                messages = [dialogue_format(image_url, prompt) for image_url in image_url_list]
                texts = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
                ii = 10
                for _ in range(ii):
                    try:
                        generated_ids = model.generate(**inputs, max_new_tokens=77)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        responses = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        break
                    except:
                        if _ == ii-1:
                            generated_ids = model.generate(**inputs, max_new_tokens=77)
                            generated_ids_trimmed = [
                                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                            ]
                            responses = processor.batch_decode(
                                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )                
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
                messages = [dialogue_format(image_url, prompt) for image_url in image_url_list]
                texts = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
                ii = 10
                for _ in range(ii):
                    try:
                        generated_ids = model.generate(**inputs, max_new_tokens=77)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        responses = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        break
                    except:
                        if _ == ii-1:
                            generated_ids = model.generate(**inputs, max_new_tokens=77)
                            generated_ids_trimmed = [
                                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                            ]
                            responses = processor.batch_decode(
                                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )            
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
