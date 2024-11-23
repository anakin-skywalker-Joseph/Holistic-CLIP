import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import re
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from PIL import Image
import json
import datetime
import sys
import logging
import ipdb
from tqdm import tqdm
from distributed import init_distributed_device

file_handler = logging.FileHandler('app.log')

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--piece-number", type=int, default=0, help="")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--batch-size",type=int,default=3)
    parser.add_argument("--numgpus", type=int, default=1)
    parser.add_argument("--input-json", type=str, default="")
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


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def get_image(image):
    if type(image) is str:
        ii = 10
        for _ in range(ii):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                if _ == ii-1:
                    raise f"Fail to read image: {image}"
                continue
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

if __name__ == "__main__":
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    rank = int(os.environ['RANK'])
    logging.info(f"rank: {rank}")
    args.horovod = False
    device = init_distributed_device(args)

    model_config = cfg.model_cfg

    model_cls = registry.get_model_class(model_config.arch)

    model = model_cls.from_config(model_config).to(device)

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device=device)
    print('Initialization Finished')
    if args.input_json == "" or not os.path.exists(args.input_json):
        input_json = 'json file path for cc12m dataset'
    else:
        input_json = args.input_json
    print(f'load from {input_json}')
    out_json = input_json[:-5]+f"_minigpt_cc12m_{rank}"+".json"
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

    if os.path.exists(out_json):
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

    prompt = 'Describe image content in English in detail:'
    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for i in tqdm(range(continue_point, len(image_pack_list))):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]

                image_list = [get_image(image) for image in image_url_list]
                chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
                question_list = [prompt]*len(image_list)
                ii = 30
                for _ in range(ii):
                    try:
                        batch_outputs = chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=77, temperature=1.0)
                        break
                    except:
                        if _ == ii-1:
                            batch_outputs = chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=77, temperature=1.0)                     
                        continue
                for j in range(len(image_url_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': batch_outputs[j]}
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
                
                image_list = [get_image(image) for image in image_url_list]
                chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
                question_list = [prompt]*len(image_list)
                ii = 30
                for _ in range(ii):
                    try:
                        batch_outputs = chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=77, temperature=0.5)
                        break
                    except:
                        if _ == ii-1:
                            batch_outputs = chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=77, temperature=0.5)
                        continue
                for j in range(len(image_url_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': batch_outputs[j]}
                    json.dump(data, outfile)
                    
                    if i==(len(image_pack_list)-1) and j==(len(image_url_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')

                print(f'{i},{len(image_pack_list)}')
                current_time = datetime.datetime.now()
                print(current_time)
                outfile.flush()