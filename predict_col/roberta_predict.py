import json
import numpy as np
from sentence_transformers import SentenceTransformer


roberta_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

def roberta_encode(inputs):
    embeds = roberta_model.encode(inputs, convert_to_tensor=True)
    return [e.tolist() for e in embeds]


def cosine_sim(e1, e2):
    e1 = np.array(e1)
    e2 = np.array(e2)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))


def predict_table_scores(file):
    data = json.load(open(file, 'r'))
    for i, item in enumerate(data):
        inputs = [item['question']] + item['tables']
        embeds = roberta_encode(inputs)
        
        data[i]['similarity'] = []
        e1 = embeds[0]
        for e2 in embeds[1: ]:
            sim = cosine_sim(e1, e2)
            data[i]['similarity'].append(sim)
    json.dump(data, open(file, 'w'), indent=2)


def predict_col_scores(file):
    data = json.load(open(file, 'r'))
    for i, item in enumerate(data):
        inputs = [item['question']] + item['columns']
        embeds = roberta_encode(inputs)
        
        data[i]['similarity'] = []
        e1 = embeds[0]
        for e2 in embeds[1: ]:
            sim = cosine_sim(e1, e2)
            data[i]['similarity'].append(sim)
    json.dump(data, open(file, 'w'), indent=2)


def evaluate_table(file, threshold):
    data = json.load(open(file, 'r'))
    correct, predict, total = 0, 0, 0
    sims_lst = []
    
    for item in data:
        for sim, label in zip(item['similarity'], item['labels']):
            if sim >= threshold:
                predict += 1
                if label == 1:
                    correct += 1
                    total += 1
                    sims_lst.append(sim)
            else:
                if label == 1:
                    total += 1
                    sims_lst.append(sim)
    print(sorted(sims_lst)[:20])
    print(correct, predict, total)
    print(correct / predict)
    print(correct / total)
    

def evaluate_col(file, threshold):
    data = json.load(open(file, 'r'))
    correct, predict, total_pos, total = 0, 0, 0, 0
    sims_lst = []
    
    for item in data:
        for sim, label in zip(item['similarity'], item['labels']):
            total += 1
            if sim >= threshold:
                predict += 1
                if label == 1:
                    correct += 1
                    total_pos += 1
                    sims_lst.append(sim)
            else:
                if label == 1:
                    total_pos += 1
                    sims_lst.append(sim)
    print(sorted(sims_lst)[:20])
    print(correct, predict, total_pos, total)
    print(correct / predict)
    print(correct / total_pos)


if __name__ == '__main__':
    predict_col_scores('../predict_col_result/predict_col_spider_test.json')
    # evaluate_col('../rlhf_result/predict_col_spider_test.json', 0.3)
