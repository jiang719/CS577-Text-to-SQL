import json
import os
import random
import re
import sys

PREDICT_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(PREDICT_DIR + '../')
from eval_utils import load_schemas_to_dict
from utils import represent_schema_text, represent_schema_code


def filter_columns(item, question, is_train=True):
    columns = []
    for c, p1, p2, p3, p4 in zip(item['columns'], item['pred_1'], item['pred_2'], item['pred_3'], item['pred_4']):
        if is_train and (p1 + p2 + p3 + p4) / 4 >= 0.08:
            columns.append(c)
        if (not is_train) and (p1 + p2 + p3 + p4) / 4 >= 0.3:
            columns.append(c)
    if (('pet' in question and 'type' in question) or ('dog' in question) or ('cat' in question)) and 'pets.pettype' in item['columns']:
        columns.append('pets.pettype')
    if ('Kyle' in question or ('name' in question and 'student' in question)) and 'highschooler.name' in item['columns']:
        columns.append('highschooler.name')
    if 'population' in question and 'country.population' in item['columns']:
        columns.append('country.population')
    if 'life expectancy' in question and 'country.lifeexpectancy' in item['columns']:
        columns.append('country.lifeexpectancy')
    if 'official' in question and 'language' in question and 'countrylanguage.isofficial' in item['columns']:
        columns.append('countrylanguage.isofficial')
    if ('Master' in question or 'Bachelor' in question) and 'degree_programs.degree_summary_name' in item['columns']:
        columns.append('degree_programs.degree_summary_name')
    if 'language' in question and ('predominant' in question or 'popular' in question) and 'countrylanguage.percentage' in item['columns']:
        columns.append('countrylanguage.percentage')
    if 'area' in question and 'country' in question and 'country.surfacearea' in item['columns']:
        columns.append('country.surfacearea')
    columns = set(columns)
    if is_train:
        columns |= set([item['columns'][i] for i in range(len(item['labels'])) if item['labels'][i] == 1])
    return columns


def merge_spider(schemas):
    for file in ('train', 'dev'):
        data = json.load(open('../dataset/spider_' + file + '.json', 'r'))
        col_data = json.load(open('spider_' + file + '_result.json', 'r'))

        for db_id in col_data:
            for item in col_data[db_id]:
                index = item['index']
                columns = filter_columns(item, data[db_id][index]['question'], file == 'train')
                data[db_id][index]['schema'] = represent_schema_code(schemas[db_id], columnset=columns, shuffle=(file=='train'))
        json.dump(data, open('../dataset/spider_' + file + '.json', 'w'), indent=2)


def merge_sparc(schemas):
    for file in ('train', 'dev'):
        data = json.load(open('../dataset/sparc_' + file + '.json', 'r'))
        col_data = json.load(open('sparc_' + file + '_result.json', 'r'))

        for db_id in col_data:
            for items in col_data[db_id]:
                columns = set()
                for item in items:
                    i1, i2 = item['index']
                    columns |= filter_columns(item, data[db_id][i1][i2]['question'], file == 'train')
                    data[db_id][i1][i2]['schema'] = represent_schema_code(schemas[db_id], columnset=columns, shuffle=(file=='train'))
        json.dump(data, open('../dataset/sparc_' + file + '.json', 'w'), indent=2)


def merge_cosql(schemas):
    for file in ('train', 'dev'):
        data = json.load(open('../dataset/cosql_' + file + '.json', 'r'))
        col_data = json.load(open('cosql_' + file + '_result.json', 'r'))

        for db_id in col_data:
            for items in col_data[db_id]:
                columns = set()
                for item in items:
                    i1, i2 = item['index']
                    columns |= filter_columns(item, data[db_id][i1][i2]['question'], file == 'train')
                    data[db_id][i1][i2]['schema'] = represent_schema_code(schemas[db_id], columnset=columns, shuffle=(file=='train'))
        json.dump(data, open('../dataset/cosql_' + file + '.json', 'w'), indent=2)


def prepare_wiki():
    data = json.load(open('/local2/jiang719/databases/sql_create_dataset/sql_create_context_v4.json', 'r'))
    random.shuffle(data)
    result = {'wiki_sql': []}
    spider_test = set()
    for test in json.load(open('../dataset/spider_dev.json', 'r')).values():
        for item in test:
            spider_test |= set(re.findall('CREATE TABLE .* \(', item['schema']))
    unique_query = set()
    unique_table = {}
    for d in data:
        overlap = False
        for test in spider_test:
            if test in d['context']:
                overlap = True
                break
        if overlap:
            continue
        if d['answer'] in unique_query:
            print(d['answer'])
            continue
        
        unique_query.add(d['answer'])
        schema = d['context']
        schema = schema.replace('VARCHAR', 'TEXT')
        schema_lst = [schema.replace('INTEGER', 'NUMBER')]
        
        if len(unique_table) > 10:
            num = random.sample([1, 2, 3], 1)[0]
            extra_schema = random.sample(list(unique_table.keys()), num)
            for name in extra_schema:
                if name not in schema_lst[0]:
                    schema_lst.append(unique_table[name])
            random.shuffle(schema_lst)
        
        result['wiki_sql'].append({
            'schema': ('; '.join(schema_lst) + ';'),
            'question': d['question'],
            'query': d['answer']
        })
        
        if schema.count('CREATE TABLE') == 1:
            table_name = schema[schema.index('CREATE TABLE') + 12: ]
            table_name = table_name[: table_name.index('(')].strip()
            unique_table[table_name] = schema
        
    json.dump(result, open('../dataset/wiki_train.json', 'w'), indent=2)


if __name__ == '__main__':
    schemas = load_schemas_to_dict('../dataset/cosql_tables.json')
    merge_spider(schemas)
    merge_sparc(schemas)
    merge_cosql(schemas)
    prepare_wiki()
