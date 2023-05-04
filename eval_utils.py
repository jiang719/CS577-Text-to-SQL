import json
import os
import re
import time
from test_gpt import generate_chatgpt
from utils import represent_schema_text


def load_data_to_dict(file):
    data = json.load(open(file, 'r'))
    result = {}
    for k, v in data.items():
        result[k] = {}
        for item in v:
            result[k][item['question']] = item['query']
    return result


def load_schemas_to_dict(file):
    schemas = json.load(open(file, 'r'))
    result = {
        schema['db_id']: schema
        for schema in schemas
    }
    return result


def get_prompt(test_db, test_question, training_data, sim_matrix, schemas, K=10):
    # prompt = 'Given a database schema, and a question, write the sql query for me. The sql must be able to be parsed into the following format:\n'
    # prompt += 'val: number(float)/string(str)/sql(dict)\n'
    # prompt += 'col_unit: (agg_id, col_id, isDistinct(bool))\n'
    # prompt += 'val_unit: (unit_op, col_unit1, col_unit2)\n'
    # prompt += 'table_unit: (table_type, col_unit/sql)\n'
    # prompt += 'cond_unit: (not_op, op_id, val_unit, val1, val2)\n'
    # prompt += "condition: [cond_unit1, 'and'/'or', cond_unit2, ...]\n"
    # prompt += 'sql {\n'
    # prompt += "  'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])\n"
    # prompt += "  'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}\n"
    # prompt += "  'where': condition\n"
    # prompt += "  'groupBy': [col_unit1, col_unit2, ...]\n"
    # prompt += "  'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])\n"
    # prompt += "  'having': condition:\n"
    # prompt += "  'limit': None/limit value\n"
    # prompt += "  'intersect': None/sql\n"
    # prompt += "  'except': None/sql\n"
    # prompt += "  'union': None/sql\n"
    # prompt += '}\n\n'
    
    # prompt += "clause_keywords = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')\n"
    # prompt += "join_keywords = ('join', 'on', 'as')\n"
    # prompt += "where_ops = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')\n"
    # prompt += "unit_ops = ('none', '-', '+', '*', '/')\n"
    # prompt += "agg_ops = ('none', 'max', 'min', 'count', 'sum', 'avg')\n"
    # prompt += "table_type = {\n"
    # prompt += "    'sql': 'sql',\n"
    # prompt += "    'table_unit': 'table_unit',\n"
    # prompt += "}\n"
    # prompt += "cond_ops = ('and', 'or')\n"
    # prompt += "sql_ops = ('intersect', 'union', 'except')\n"
    # prompt += "order_ops = ('desc', 'asc')\n\n"
    prompt = 'Given a database schema, and a question, write the sql query for me.\n\n'
    
    shot_db = sim_matrix[test_db][test_question][0][0]
    prompt += 'Database schema:\n' + represent_schema_text(schemas[shot_db])
    prompt += '###\n'
    for shot in sim_matrix[test_db][test_question][: K]:
        if shot[0] != shot_db:
            shot_db = shot[0]
            prompt += '\n'
            prompt += 'Database schema:\n' + represent_schema_text(schemas[shot_db]) + '###\n'
        prompt += 'Question: ' + shot[1] + '\n'
        prompt += 'Query: ' + training_data[shot_db][shot[1]] + '\n###\n'
    prompt += '\n'
    
    prompt += 'Database schema:\n' + represent_schema_text(schemas[test_db])
    prompt += '###\n'
    prompt += 'Question: ' + test_question + '\n'
    
    return prompt


def run_spider_test():
    training_data = load_data_to_dict('dataset/spider_train.json')
    testing_data = load_data_to_dict('dataset/spider_dev.json')
    sim_matrix = json.load(open('fsl/text_roberta/spider_dev_question_similarity.json', 'r'))
    schemas = load_schemas_to_dict('dataset/spider_tables.json')
    
    if os.path.exists('fsl/text_roberta/spider_dev_result.json'):
        result = json.load(open('fsl/text_roberta/spider_dev_result.json', 'r'))
    else:
        result = {}
    finished = {k: set([v['question'] for v in result[k]]) for k in result}
    failed_cnt = 0
    for test_db in testing_data:
        if test_db not in result:
            result[test_db] = []
            finished[test_db] = set()
        for test_question in testing_data[test_db]:
            if test_question in finished[test_db]:
                continue
            prompt = get_prompt(test_db, test_question, training_data, sim_matrix, schemas, K=20)
            try:
                response = generate_chatgpt(prompt, max_tokens=128)
                query = response[response.index('Query: ') + 7: ].strip()
                
                result[test_db].append({
                    'question': test_question, 'query': query
                })
                finished[test_db].add(test_question)
                
                print(test_db, '\t', test_question, '\t failed_cnt: {}'.format(failed_cnt))
                print(response)
            except Exception as e:
                print(str(e))
                failed_cnt += 1
            time.sleep(3)
            json.dump(result, open('fsl/text_roberta/spider_dev_result.json', 'w'), indent=2)


def run_spider_train():
    training_data = load_data_to_dict('dataset/spider_train.json')
    sim_matrix = json.load(open('fsl/text_roberta/spider_train_question_similarity.json', 'r'))
    schemas = load_schemas_to_dict('dataset/spider_tables.json')
    
    if os.path.exists('fsl/text_roberta/spider_train_result.json'):
        result = json.load(open('fsl/text_roberta/spider_train_result.json', 'r'))
    else:
        result = {}
    finished = {k: set([v['question'] for v in result[k]]) for k in result}
    failed_cnt = 0
    for train_db in training_data:
        if train_db not in result:
            result[train_db] = []
            finished[train_db] = set()
        for train_question in training_data[train_db]:
            if train_question in finished[train_db]:
                continue
            prompt = get_prompt(train_db, train_question, training_data, sim_matrix, schemas, K=8)
            response = generate_chatgpt(prompt, max_tokens=128)
            try:
                query = response[response.index('Query: ') + 7: ].strip()
                
                result[train_db].append({
                    'question': train_question, 'query': query
                })
                finished[train_db].add(train_question)
                
                print(train_db, '\t', train_question, '\t failed_cnt: {}'.format(failed_cnt))
            except Exception as e:
                print(str(e))
                print(response)
                failed_cnt += 1
            time.sleep(1)
            json.dump(result, open('fsl/text_roberta/spider_train_result.json', 'w'), indent=2)


def convert_execution_format_spider(gt_file, result_file, gt_output, result_output):
    gt = json.load(open(gt_file, 'r'))
    result = json.load(open(result_file, 'r'))
    result = {
        k: {v['question']: v['query']
            for v in result[k]} for k in result
    }
    
    w1 = open(gt_output, 'w')
    w2 = open(result_output, 'w')
    cnt = 0
    for db_id in gt:
        for i, item in enumerate(gt[db_id]):
            if db_id in result and item['question'] in result[db_id]:
                if type(result[db_id][item['question']]) == list:
                    pred = result[db_id][item['question']][0].strip()
                else:
                    pred = result[db_id][item['question']].strip()
                w2.write(re.sub('\\s+', ' ', pred) + '\n')
                w1.write(re.sub('\\s+', ' ', item['query']) + '\t' + db_id + '\n')
                cnt += 1
                # if cnt == 33:
                #     return
            else:
                pass
    w1.close()
    w2.close()


def convert_execution_format_sparc(gt_file, result_file, gt_output, result_output):
    gt = json.load(open(gt_file, 'r'))
    result = json.load(open(result_file, 'r'))
    
    w1 = open(gt_output, 'w')
    w2 = open(result_output, 'w')
    cnt = 0
    for db_id in gt:
        if db_id not in result:
            continue
        for gt_iter, pred_iter in zip(gt[db_id], result[db_id]):
            min_len = min([len(gt_iter), len(pred_iter)])
            gt_iter = gt_iter[: min_len]
            pred_iter = pred_iter[: min_len]
            for gt_item, pred_item in zip(gt_iter, pred_iter):
                cnt += 1
                gt_sql = re.sub('\\s+', ' ', gt_item['query']).strip()
                if type(pred_item['query']) == list:
                    pred = re.sub('\\s+', ' ', pred_item['query'][0]).strip()
                else:
                    pred = re.sub('\\s+', ' ', pred_item['query']).strip()
                w2.write(pred + '\n')
                w1.write(gt_sql + '\t' + db_id + '\n')
            w1.write('\n')
            w2.write('\n')
    w1.close()
    w2.close()


# python evaluation.py --gold /local2/jiang719/databases/spider/gt.sql --pred /local2/jiang719/databases/spider/pred.sql --etype all --db /local2/jiang719/databases/spider/database --table /local2/jiang719/databases/spider/tables.json

if __name__ == '__main__':
    # convert_execution_format_spider(
    #     gt_file='dataset/spider_dev.json',
    #     result_file='finetuning_rlhf_result/spider_dev_raw_llama.json',
    #     # result_file='fsl_result/text_roberta/spider_dev_result.json',
    #     gt_output='/local2/jiang719/databases/spider/gt.sql',
    #     result_output='/local2/jiang719/databases/spider/pred.sql',
    # )
    convert_execution_format_sparc(
        gt_file='dataset/cosql_dev.json',
        result_file='finetuning_rlhf_result/cosql_dev_col_result_codegen_2b.json',
        gt_output='/local2/jiang719/databases/sparc/gt.sql',
        result_output='/local2/jiang719/databases/sparc/pred.sql',
    )
