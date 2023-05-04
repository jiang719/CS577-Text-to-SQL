import json
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import sys

PREDICT_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(PREDICT_DIR + '../')
from eval_utils import load_schemas_to_dict
from utils import represent_schema


def get_primary_foreign_keys(schemas, db_id):
    tables, primary, foreign = represent_schema(schemas[db_id])
    primary = [
        (k + '.' + v).lower() for k, v in primary.items()
    ]
    foreign = {
        (k + '.' + v).lower(): (foreign[k][v][0] + '.' + foreign[k][v][1]).lower()for k in foreign for v in foreign[k]
    }
    return primary, foreign


def eval(predicts, labels, th):
    correct, predict, total_pos, total = 0, 0, 0, 0
    for p, gt in zip(predicts, labels):
        if gt < 0:
            continue
        total += 1
        if p >= th:
            predict += 1
            if gt == 1:
                correct += 1
                total_pos += 1
        else:
            if gt == 1:
                total_pos += 1
    print(correct, predict, total_pos, total)
    print(correct / predict, correct / total_pos, predict / total)


data = json.load(open('../dataset/spider_dev.json', 'r'))
result = json.load(open('../predict_col_result/spider_dev_result.json', 'r'))
model_prediction, ground_truth = [], []
for db_id in result:
    for item in result[db_id]:
        # for item in items:
        lst = np.array(item['pred_1']) + np.array(item['pred_2']) + np.array(item['pred_3']) + np.array(item['pred_4']) / 4
        question = data[db_id][item['index']]['question']
        columns = item['columns']
        # if (('pet' in question and 'type' in question) or ('dog' in question) or ('cat' in question)) and 'pets.pettype' in columns:
        #     lst[columns.index('pets.pettype')] = 1
        # if ('Kyle' in question or ('name' in question and 'student' in question)) and 'highschooler.name' in columns:
        #     lst[columns.index('highschooler.name')] = 1
        # if 'population' in question and 'country.population' in columns:
        #     lst[columns.index('country.population')] = 1
        # if 'life expectancy' in question and 'country.lifeexpectancy' in columns:
        #     lst[columns.index('country.lifeexpectancy')] = 1
        # if 'official' in question and 'language' in question and 'countrylanguage.isofficial' in columns:
        #     lst[columns.index('countrylanguage.isofficial')] = 1
        # if ('Master' in question or 'Bachelor' in question) and 'degree_programs.degree_summary_name' in columns:
        #     lst[columns.index('degree_programs.degree_summary_name')] = 1
        # if 'language' in question and ('predominant' in question or 'popular' in question) and 'countrylanguage.percentage' in columns:
        #     lst[columns.index('countrylanguage.percentage')] = 1
        # if 'area' in question and 'country' in question and 'country.surfacearea' in columns:
        #     lst[columns.index('country.surfacearea')] = 1
    
        model_prediction += lst.tolist()
        ground_truth += item['labels']


eval(model_prediction, ground_truth, 0.1)

fpr, tpr, thresholds = metrics.roc_curve(ground_truth, model_prediction)
roc_auc = metrics.auc(fpr, tpr)

print('auc:', roc_auc)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, 'b', label = 'Test AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right', fontsize=16)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
plt.ylabel('True Positive Rate', fontsize=18)
plt.xlabel('False Positive Rate', fontsize=18)
plt.tight_layout()
plt.savefig('dev_auc.pdf', bbox_inches='tight')
