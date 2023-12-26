#! /usr/bin/env python

import json
from collections import Counter
from argparse import ArgumentParser
import os

# parser = ArgumentParser()
# parser.add_argument("--path", type=str, required=True)
#
# args = parser.parse_args()

with open("//data/output.json") as f:
    # data = [json.loads(line) for line in f]
    data = []
    line = f.read(1)
    print(line)
    while line:
        if line == "{":
            j = ""
            while line != "}":
                if line != '\n':
                    j += line
                line = f.read(1)
            j += "}"
            print(j)
            data += [json.loads(j)]
        line = f.read(1)
for x in data:
    print(x['output'])
train_examples = [{
    "prompt": x['instruction']+x['input'],
    "response": x['output']
} for x in data]

os.makedirs("../formatted_data", exist_ok=True)

with open("../formatted_data/output.jsonl", "w") as f:
    for e in train_examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
