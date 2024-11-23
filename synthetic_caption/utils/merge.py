import json
import jsonlines
import os
from tqdm import tqdm
import ipdb
from collections import OrderedDict

dicts = {}
numgpus = 16 # set up the number of splitted files
for i in tqdm(range(numgpus)):
    with open(f"your output file{i}.json", "r") as f:
        data = json.load(f)
        for d in data:
            dicts[d["image"]] = d["caption"]
with jsonlines.open(f"composed.jsonl", "w") as writer:
    for key,value in tqdm(dicts.items()):
        lines = {"image": key, "caption": value}
        writer.write(lines)
with open(f"composed_index.json", "w") as writer:
    json.dump(dicts, writer)
