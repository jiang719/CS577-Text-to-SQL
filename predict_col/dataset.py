import json
import torch
import random
import numpy as np

random.seed(7)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, shuffle=False):
        self.data = []

        dataset = json.load(open(file_path, 'r'))
        for sample in dataset:
            question = sample['question'].lower()
            raw_columns, columns = [], []
            # too_long = False
            for c in sample['columns']:
                col = tokenizer(question, c.lower())
                if len(col['input_ids']) >= 512:
                    for k in col:
                        col[k] = col[k][-512: ]
                    # too_long = True
                    # break
                columns.append(col)
                raw_columns.append(c.lower())
            # if too_long:
            #     continue
            
            self.data.append({
                'db_id': sample['db_id'],
                'index': sample['index'],
                # 'raw_question': question,
                # 'raw_columns': raw_columns,
                'columns': columns,
                'adj': sample['adj'],
                'labels': sample['labels']
            })
        
        if shuffle:
            random.shuffle(self.data)
        
        print(file_path, 'loaded:', len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    def merge(self, dataset, shuffle=False):
        self.data = self.data + dataset.data
        if shuffle:
            random.shuffle(self.data)


def custom_collate(batch):
    max_c_num = max([len(b['columns']) for b in batch])
    max_c_len = max([
        max([len(c['input_ids']) for c in b['columns']]) for b in batch
    ])
    batch_data = {
        'columns': {
            'input_ids': np.zeros((len(batch), max_c_num, max_c_len)),
            'token_type_ids': np.zeros((len(batch), max_c_num, max_c_len)),
            'attention_mask': np.zeros((len(batch), max_c_num, max_c_len))
        }, 
        'adj': np.zeros((len(batch), max_c_num, max_c_num)), 'labels': []}
    for i, b in enumerate(batch):
        for j, c in enumerate(b['columns']):
            l = len(c['input_ids'])
            batch_data['columns']['input_ids'][i, j, :l] = np.array(c['input_ids'])
            batch_data['columns']['token_type_ids'][i, j, :l] = np.array(c['token_type_ids'])
            batch_data['columns']['attention_mask'][i, j, :l] = np.array(c['attention_mask'])
        batch_data['adj'][i, :len(b['columns']), :len(b['columns'])] = np.array(b['adj'])
        batch_data['labels'].append(b['labels'] + [-100]*(max_c_num - len(b['labels'])))
    for k in batch_data:
        if k == 'columns':
            for kk in batch_data[k]:
                batch_data[k][kk] = torch.LongTensor(batch_data[k][kk])
        else:
            batch_data[k] = torch.LongTensor(batch_data[k])
    return batch_data
