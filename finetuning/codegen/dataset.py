import json
import torch
import codecs
import random

random.seed(7)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048, max_cnt=None, shuffle=False):
        self.data = []
        self.max_length = max_length

        fp = codecs.open(file_path, 'r', 'utf-8')
        for l in fp.readlines():
            l = json.loads(l)

            inputs = 'Schema: ' + l['schema'].strip() + '\nQuestion: ' + l['question'].strip() + \
                '\nQuery: ' + l['query'].strip() + tokenizer.eos_token
            outputs = l['query'].strip() + tokenizer.eos_token
            inputs = tokenizer.encode(inputs, return_tensors='pt')
            outputs = tokenizer.encode(outputs, return_tensors='pt')
            if inputs.size(1) > max_length:
                continue
            self.data.append({
                'input_ids': inputs,
                'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
                'attention_mask': torch.ones(inputs.size()).long()
            })
        
        if shuffle:
            random.shuffle(self.data)
        if max_cnt is not None:
            self.data = self.data[: max_cnt]
        # self.data = self.data[::-1]
        print(file_path, 'total size:', len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    def merge(self, dataset, cnt=None, shuffle=False):
        if cnt is not None:
            self.data += dataset.data[: cnt]
        else:
            self.data += dataset.data
        if shuffle:
            random.shuffle(self.data)


def custom_collate(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_len = max([b['input_ids'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_len - b['input_ids'].size(1)).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data
