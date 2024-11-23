import jsonlines
import os
import sys
from tqdm import tqdm
import json

def read_jsonl(file_path):
    with jsonlines.open(file_path) as reader:
        data = {line for line in reader}
    return data

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"successfully read json file {file_path}")
    return data

def save_jsonl(data, file_path):
    with jsonlines.open(file_path, 'w') as writer:
        for line in data:
            writer.write(line)

if __name__ == '__main__':
    input_file = "original.jsonl" #original jsonl file that is in the format of {"image": image_path, "caption": original_caption}
    output_file = 'output_file.jsonl'
    internvl_file_detail_name = 'detail_index.json'
    internvl_file_noun_name = 'noun_index.json'
    internvl_file_main_name = "main_index.json"
    internvl_file_style_name = "style_index.json"
    internvl_file_background_name = "background_index.json"

    internvl_detail = read_json(internvl_file_detail_name)
    internvl_noun = read_json(internvl_file_noun_name)
    internvl_main = read_json(internvl_file_main_name)
    internvl_background = read_json(internvl_file_background_name)
    internvl_style = read_json(internvl_file_style_name)

    with jsonlines.open(input_file, 'r') as f:
        with jsonlines.open(output_file,'w') as writer:
            for line in tqdm(f):
                image_path = line["image"]
                caption = line["caption"]
                caption_internvl_detail = internvl[image_path]
                caption_internvl_noun = internvl_noun[image_path]
                caption_internvl_main = internvl_main[image_path]
                caption_internvl_background = internvl_background[image_path]
                caption_internvl_style = internvl_style[image_path]
                new_line = {"image": image_path, "caption": [caption,caption_internvl_detail,caption_internvl_noun,caption_internvl_background,caption_internvl_main,caption_internvl_style]}
                writer.write(new_line)