import json
import numpy as np
from evaluation import eval_hardness
from utils import represent_schema_text
from sentence_transformers import SentenceTransformer


roberta_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

def roberta_encode(input):
    embeds = roberta_model.encode(input, convert_to_tensor=True)
    return [e.tolist() for e in embeds]


def cosine_sim(e1, e2):
    e1 = np.array(e1)
    e2 = np.array(e2)
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))


def dev_database_similarity():
    train = json.load(open('dataset/spider_train.json', 'r'))
    train = list(train.keys())
    dev = json.load(open('dataset/spider_dev.json', 'r'))
    dev = list(dev.keys())
    
    data = json.load(open('dataset/spider_tables.json', 'r'))
    texts = [
        represent_schema_text(d) for d in data
    ]
    embeds = roberta_encode(texts)
    embeds = {
        data[i]['db_id']: embeds[i] for i in range(len(data))
    }
    result = {}
    for d in dev:
        e1 = embeds[d]
        sims = []
        for t in train:
            e2 = embeds[t]
            sims.append([t, cosine_sim(e1, e2)])
        sims = sorted(sims, key=lambda e: e[1], reverse=True)
        result[d] = sims[: 5]
    json.dump(result, open('fsl/text_roberta/spider_dev_database_similarity.json', 'w'), indent=2)


def train_database_similarity():
    train = json.load(open('dataset/spider_train.json', 'r'))
    train = list(train.keys())
    data = json.load(open('dataset/spider_tables.json', 'r'))
    data = [d for d in data if d['db_id'] in train]
    texts = [
        represent_schema_text(d) for d in data
    ]
    embeds = roberta_encode(texts)
    embeds = {
        data[i]['db_id']: embeds[i] for i in range(len(data))
    }
    result = {}
    for d in train:
        e1 = embeds[d]
        sims = []
        for t in train:
            if t == d:
                continue
            e2 = embeds[t]
            sims.append([t, cosine_sim(e1, e2)])
        sims = sorted(sims, key=lambda e: e[1], reverse=True)
        result[d] = sims[: 5]
    json.dump(result, open('fsl/text_roberta/spider_train_database_similarity.json', 'w'), indent=2)


def dev_question_similarity():
    train = json.load(open('dataset/spider_train.json', 'r'))
    dev = json.load(open('dataset/spider_dev.json', 'r'))
    db_sim = json.load(open('fsl/text_roberta/spider_dev_database_similarity.json', 'r'))
    result = {}
    for d_id in dev:
        print(d_id)
        result[d_id] = {}
        for t_id in db_sim[d_id]:
            t_id, t_sim = t_id
            questions = [d['question'] for d in dev[d_id]] + [d['question'] for d in train[t_id]]
            embeds = roberta_encode(questions)
            for i in range(len(dev[d_id])):
                sims = []
                for j in range(len(train[t_id])):
                    e1 = embeds[i]
                    e2 = embeds[len(dev[d_id]) + j]
                    sims.append([t_id, train[t_id][j]['question'], cosine_sim(e1, e2), eval_hardness(train[t_id][j]['sql'])])
                sims = sorted(sims, key=lambda e: e[2], reverse=True)
                if dev[d_id][i]['question'] not in result[d_id]:
                    result[d_id][dev[d_id][i]['question']] = []
                easy, medium, hard, extra = 0, 0, 0, 0
                for item in sims:
                    if item[3] == 'easy' and easy < 1:
                        easy += 1
                        result[d_id][dev[d_id][i]['question']].append(item)
                    if item[3] == 'medium' and medium < 1:
                        medium += 1
                        result[d_id][dev[d_id][i]['question']].append(item)
                    if item[3] == 'hard' and hard < 1:
                        hard += 1
                        result[d_id][dev[d_id][i]['question']].append(item)
                    if item[3] == 'extra' and extra < 1:
                        extra += 1
                        result[d_id][dev[d_id][i]['question']].append(item)
                    if easy == medium == hard == extra == 1:
                        break
        json.dump(result, open('fsl/text_roberta/spider_dev_question_similarity.json', 'w'), indent=2)


def train_question_similarity():
    train = json.load(open('dataset/spider_train.json', 'r'))
    db_sim = json.load(open('fsl/text_roberta/spider_train_database_similarity.json', 'r'))
    result = {}
    for d_id in train:
        print(d_id)
        result[d_id] = {}
        for t_id in db_sim[d_id][: 2]:
            t_id, t_sim = t_id
            questions = [d['question'] for d in train[d_id]] + [d['question'] for d in train[t_id]]
            embeds = roberta_encode(questions)
            for i in range(len(train[d_id])):
                sims = []
                for j in range(len(train[t_id])):
                    e1 = embeds[i]
                    e2 = embeds[len(train[d_id]) + j]
                    sims.append([t_id, train[t_id][j]['question'], cosine_sim(e1, e2), eval_hardness(train[t_id][j]['sql'])])
                sims = sorted(sims, key=lambda e: e[2], reverse=True)
                if train[d_id][i]['question'] not in result[d_id]:
                    result[d_id][train[d_id][i]['question']] = []
                easy, medium, hard, extra = 0, 0, 0, 0
                for item in sims:
                    if item[3] == 'easy' and easy < 1:
                        easy += 1
                        result[d_id][train[d_id][i]['question']].append(item)
                    if item[3] == 'medium' and medium < 1:
                        medium += 1
                        result[d_id][train[d_id][i]['question']].append(item)
                    if item[3] == 'hard' and hard < 1:
                        hard += 1
                        result[d_id][train[d_id][i]['question']].append(item)
                    if item[3] == 'extra' and extra < 1:
                        extra += 1
                        result[d_id][train[d_id][i]['question']].append(item)
                    if easy == medium == hard == extra == 1:
                        break
        json.dump(result, open('fsl/text_roberta/spider_train_question_similarity.json', 'w'), indent=2)


if __name__ == '__main__':
    # train_database_similarity()
    train_question_similarity()
