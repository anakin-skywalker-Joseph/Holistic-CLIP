import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
import datetime
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import ipdb
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.multiprocessing as mp
import logging
from distributed import init_distributed_device

def eval_model(args, split_num = 4):
    rank = int(os.environ['RANK'])
    logging.info(f"rank: {rank}")
    args.horovod = False
    device = init_distributed_device(args)
    # device = torch.device(f'cuda:{rank}')
    # disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model = model.to(device)
    print('load model')
    if args.input_json == "" or not os.path.exists(args.input_json):
        input_json = 'json file path for cc12m dataset'
    else:
        input_json = args.input_json
    print(input_json)
    out_json = input_json[:-5]+f"_llava13b_cc12m_{rank}"+".json"
    
    image_list = []
    image_pack_list = []
    number=0
    batch_size = args.batch_size
    with open(input_json, 'r') as json_file:
        data = json.load(json_file)
    data_length = len(data)
    split_length = data_length//split_num
    start = rank*split_length
    end = (rank+1)*split_length if rank!=split_num-1 else data_length
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
        continue_point=0
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
    prompt = 'Describe the <image> in English in detail:'
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    if os.path.exists(out_json):
        with open(out_json, 'a') as outfile:
            for i in tqdm(range(continue_point,len(image_pack_list))):
                if i == 0:
                    outfile.write('[')
                image_url_list = image_pack_list[i]
                image_list = []
                for img_path in image_url_list:
                    try:
                        image = image_processor.preprocess(Image.open(img_path), return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device)
                        image_list.append(image)
                    except Exception as e:
                        print(f"Error processing image at path {img_path}: {str(e)}")
                        continue
                new_input_ids = input_ids.repeat(len(image_list), 1)
                with torch.inference_mode():
                    output_ids = model.generate(
                        new_input_ids,
                        images=image_list,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=77,
                        use_cache=True)
                caption = []
                for _ in output_ids:
                    caption.append(tokenizer.decode(_.cpu(),
                                            skip_special_tokens=True).strip().replace('\"',' '))
                for j in range(len(image_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': caption[j]}
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
                image_list = []
                for img_path in image_url_list:
                    try:
                        image = image_processor.preprocess(Image.open(img_path), return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device)
                        image_list.append(image)
                    except Exception as e:
                        print(f"Error processing image at path {img_path}: {str(e)}")
                        continue
                new_input_ids = input_ids.repeat(len(image_list), 1)
                with torch.inference_mode():
                    output_ids = model.generate(
                        new_input_ids,
                        images=image_list,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=77,
                        use_cache=True)
                caption = []
                for _ in output_ids:
                    caption.append(tokenizer.decode(_.cpu(),
                                            skip_special_tokens=True).strip().replace('\"',' '))
                for j in range(len(image_list)):
                    image_url = image_url_list[j]
                    data = {'image': image_url, 'caption': caption[j]}
                    json.dump(data, outfile)
                    
                    if i==(len(image_pack_list)-1) and j==(len(image_url_list)-1):
                        outfile.write(']')
                    else:
                        outfile.write(',')

                print(f'{i},{len(image_pack_list)}')
                current_time = datetime.datetime.now()
                print(current_time)
                outfile.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--piece-number", type=int, default=0, help="")
    parser.add_argument("--batch-size",type=int,default=1)
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
    eval_model(args = args, split_num=args.numgpus)
