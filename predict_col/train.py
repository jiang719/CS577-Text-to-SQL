import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset, custom_collate
from transformers import AutoTokenizer, BertForNextSentencePrediction, Adafactor
from rgcn import RGCN


def eval(outputs, labels, th):
    predicts = torch.exp(outputs)
    correct, predict, total_pos, total = 0, 0, 0, 0
    min_score = 10
    for p, gt in zip(predicts, labels):
        if gt < 0:
            continue
        p = p[1]
        total += 1
        if p >= th:
            predict += 1
            if gt == 1:
                correct += 1
                total_pos += 1
        else:
            if gt == 1:
                total_pos += 1
        if gt == 1 and p < min_score:
            min_score = p
    return correct, predict, total_pos, total, float(min_score)


def validate(validation_loader, model, save_path):
    model.eval()
    torch.cuda.empty_cache()
    loss_fct = nn.NLLLoss()
    validation_loss = []
    prediction, ground_truth = [], []
    validate_correct, validate_predict, validate_total_pos, validate_total = 0, 0, 0, 0
    validate_min_score = 1
    oom = 0
    for data in validation_loader:
        data = {
            'columns': {k: data['columns'][k].cuda() for k in data['columns']},
            'adj': data['adj'].cuda(),
            'labels': data['labels'].cuda()
        }
        with torch.no_grad():
            try:
                outputs = model(data).view(-1, 2)
                labels = data['labels'].view(-1)
                
                prediction += outputs.tolist()
                ground_truth += labels.tolist()
                
                loss = loss_fct(outputs, labels)
                correct, predict, total_pos, total, min_score = eval(outputs, labels, 0.5)
                validation_loss.append(loss.mean().item())
                validate_correct += correct
                validate_predict += predict
                validate_total_pos += total_pos
                validate_total += total
                validate_min_score = min(validate_min_score, min_score)
            except:
                oom += 1
                torch.cuda.empty_cache()
                del data
    print('validation loss: {}, precision: {}, recall: {}, percentage: {}, min_score: {}, oom: {}'.format(
        round(np.mean(validation_loss), 6),
        round(validate_correct / (validate_predict + 1e-4), 4),
        round(validate_correct / validate_total_pos, 4),
        round(validate_predict / validate_total, 4),
        round(validate_min_score, 4), oom
    ))
    torch.save({
        'model': model.state_dict(),
        'prediction': prediction,
        'ground_truth': ground_truth
    }, save_path)
    model.train()


def train(training_loader, validation_loader, epoches, save_path):
    # scaler = torch.cuda.amp.GradScaler()
    def sampling(outputs, labels):
        index = (labels == 0).nonzero().squeeze(-1)
        negative = labels.index_select(0, index)
        output_neg = outputs.index_select(0, index)
        index = (labels == 1).nonzero().squeeze(-1)
        positive = labels.index_select(0, index)
        output_pos = outputs.index_select(0, index)
        
        if positive.size(0) >= negative.size(0):
            return outputs, labels

        index = torch.LongTensor(random.sample([i for i in range(negative.size(0))], positive.size(0))).cuda()

        # labels = torch.cat([negative, positive] + [positive for _ in range(len(negative) // len(positive) // 4)], dim=0)
        # outputs = torch.cat([output_neg, output_pos] + [output_pos for _ in range(len(negative) // len(positive) // 4)], dim=0)
        labels = torch.cat([negative.index_select(0, index), positive], dim=0)
        outputs = torch.cat([output_neg.index_select(0, index), output_pos], dim=0)
        return outputs, labels
        
    optimizer = Adafactor(model.parameters(), lr=2.5e-5, scale_parameter=False, relative_step=False)
    loss_fct = nn.NLLLoss()
    for epoch in range(epoches):
        model.train()
        training_loss = []
        train_correct, train_predict, train_total_pos, train_total = 0, 0, 0, 0
        train_min_score = 1
        oom = 0
        for i, data in enumerate(training_loader):
            data = {
                'columns': {k: data['columns'][k].cuda() for k in data['columns']},
                'adj': data['adj'].cuda(),
                'labels': data['labels'].cuda()
            }
            try:
                outputs = model(data)
                outputs = outputs.view(-1, 2)
                labels = data['labels'].view(-1)
                
                correct, predict, total_pos, total, min_score = eval(outputs, labels, 0.5)
                outputs, labels = sampling(outputs, labels)
                
                loss = loss_fct(outputs, labels)
                optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                optimizer.step()
                training_loss.append(loss.mean().item())
                
                train_correct += correct
                train_predict += predict
                train_total_pos += total_pos
                train_total += total
                train_min_score = min(train_min_score, min_score)
            except:
                oom += 1
                del data
                try:
                    del outputs, logits
                except:
                    pass
                torch.cuda.empty_cache()
                continue
            
            if i % 1000 == 0 and i > 0:
                print('epoch: {}, step: {}/{}, loss: {}, precision: {}, recall: {}, percentage: {}, min_score: {}, oom: {}'.format(
                    epoch + 1, i, len(training_loader), 
                    round(np.mean(training_loss), 3),
                    round(train_correct / (train_predict + 1e-4), 3),
                    round(train_correct / train_total_pos, 3),
                    round(train_predict / train_total, 3),
                    round(train_min_score, 4), oom
                ))
                oom = 0
                training_loss = []
                train_correct, train_predict, train_total_pos, train_total = 0, 0, 0, 0
                torch.cuda.empty_cache()
            if i % 2500 == 0 and i > 0:  
                validate(validation_loader, model, save_path + 'rgcn_3_{}.bin'.format(epoch + 1))
        validate(validation_loader, model, save_path + 'rgcn_3_{}.bin'.format(epoch + 1))


if __name__ == '__main__':
    device_map = {'bert.embeddings': 0, 'bert.pooler': 0, 'cls': 0}
    device_map.update({'bert.encoder.layer.' + str(i): 0 for i in range(0, 3)})
    device_map.update({'bert.encoder.layer.' + str(i): 1 for i in range(3, 10)})
    device_map.update({'bert.encoder.layer.' + str(i): 2 for i in range(10, 17)})
    device_map.update({'bert.encoder.layer.' + str(i): 3 for i in range(17, 24)})

    tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
    encoder = BertForNextSentencePrediction.from_pretrained('bert-large-uncased', device_map=device_map)
    model = RGCN(encoder, 4, num_layers=4, dropout=0.1, device=0)
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    
    training_files = [
        '../predict_col_result/predict_col_spider_train.json',
        '../predict_col_result/predict_col_cosql_train.json',
        '../predict_col_result/predict_col_sparc_train.json'
    ]
    validation_files = [
        '../predict_col_result/predict_col_spider_test.json',
        '../predict_col_result/predict_col_sparc_test.json',
        '../predict_col_result/predict_col_cosql_test.json'
    ]
    training_dataset = Dataset(training_files[0], tokenizer)
    for file in training_files[1:]:
        training_dataset.merge(Dataset(file, tokenizer), shuffle=True)
    validation_dataset = Dataset(validation_files[0], tokenizer)
    for file in validation_files[1:]:
        validation_dataset.merge(Dataset(file, tokenizer))
    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )
    train(training_loader, validation_loader, 3, '../')
