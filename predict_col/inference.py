import json
import re
import torch
from dataset import Dataset, custom_collate
from transformers import AutoTokenizer, BertForNextSentencePrediction
from rgcn import RGCN


def load_dataset(files):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    validation_dataset = Dataset(validation_files[0], tokenizer)
    for file in validation_files[1:]:
        validation_dataset.merge(Dataset(file, tokenizer))
    return validation_dataset


def inference_spider(model, validation_dataset, validation_loader):
    model.eval()
    # result = {}
    result = json.load(open('../predict_col_result/spider_train_result.json', 'r'))
    for i, data in enumerate(validation_loader):
        inputs = {
            'columns': {k: torch.LongTensor(data['columns'][k]).cuda() for k in data['columns']},
            'adj': torch.LongTensor(data['adj']).cuda(),
            'labels': torch.LongTensor(data['labels']).cuda()
        }
        db_id = validation_dataset.data[i]['db_id']
        index = validation_dataset.data[i]['index']
        
        if db_id not in result:
            print(db_id)
            result[db_id] = []
        with torch.no_grad():
            outputs = model(inputs).view(-1, 2)
            labels = inputs['labels'].view(-1)
            pred = torch.exp(outputs)[:, 1].tolist()
            labels = labels.tolist()
            # result[db_id].append({
            #     'index': validation_dataset.data[i]['index'],
            #     'columns': validation_dataset.data[i]['raw_columns'],
            #     'pred_1': pred,
            #     'labels': labels
            # })
            assert len(pred) == len(result[db_id][index]['pred_1'])
            result[db_id][index]['pred_4'] = pred
    json.dump(result, open('../predict_col_result/spider_train_result.json', 'w'), indent=2) 


def inference_multiturn(model, validation_dataset, validation_loader):
    model.eval()
    # result = {}
    result = json.load(open('../predict_col_result/cosql_train_result.json', 'r'))
    for i, data in enumerate(validation_loader):
        inputs = {
            'columns': {k: torch.LongTensor(data['columns'][k]).cuda() for k in data['columns']},
            'adj': torch.LongTensor(data['adj']).cuda(),
            'labels': torch.LongTensor(data['labels']).cuda()
        }
        db_id = validation_dataset.data[i]['db_id']
        iter, index = validation_dataset.data[i]['index']
        
        if db_id not in result:
            print(db_id)
            result[db_id] = []
        if index == 0:
            result[db_id].append([])
        with torch.no_grad():
            outputs = model(inputs).view(-1, 2)
            labels = inputs['labels'].view(-1)
            pred = torch.exp(outputs)[:, 1].tolist()
            labels = labels.tolist()
            # result[db_id][iter].append({
            #     'index': validation_dataset.data[i]['index'],
            #     'columns': validation_dataset.data[i]['raw_columns'],
            #     'pred_1': pred,
            #     'labels': labels
            # })
            assert len(pred) == len(result[db_id][iter][index]['pred_1'])
            result[db_id][iter][index]['pred_4'] = pred
    json.dump(result, open('../predict_col_result/cosql_train_result.json', 'w'), indent=2)            


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
    encoder = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
    model = RGCN(encoder, 4, num_layers=4, dropout=0.1, device=0).cuda()
    load = torch.load('../rgcn_4.bin')
    model.load_state_dict(load['model'])
    
    validation_files = [
        # '../predict_col_result/predict_col_spider_train.json',
        # '../predict_col_result/predict_col_sparc_train.json',
        '../predict_col_result/predict_col_cosql_train.json'
    ]
    validation_dataset = load_dataset(validation_files)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )
    
    inference_multiturn(model, validation_dataset, validation_loader)
