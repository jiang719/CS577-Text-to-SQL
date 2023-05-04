import json
import torch
import codecs
import random

random.seed(7)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048, shuffle=False):
        self.data = []
        self.max_length = max_length

        fp = codecs.open(file_path, 'r', 'utf-8')
        for l in fp.readlines():
            l = json.loads(l)

            if len(l['query'].split()) < 30:
                continue

            inputs = 'Schema: ' + l['schema'] + '\nQuestion: ' + l['question'] + '\nQuery: '
            outputs = inputs + l['query'] + '<|endofmask|>'
            inputs = tokenizer(inputs, return_tensors="pt").input_ids
            outputs = tokenizer(outputs, return_tensors="pt").input_ids
            if outputs.size(1) > max_length:
                continue
            self.data.append({
                'db_name': l['db_name'],
                'input_ids': inputs,
                'output_ids': outputs
            })
        
        if shuffle:
            random.shuffle(self.data)

        print(file_path, 'total size:', len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    def merge(self, dataset, shuffle=False):
        self.data += dataset.data
        if shuffle:
            random.shuffle(self.data)