import json
import os
import re
import sys

PREDICT_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(PREDICT_DIR + '../')
from eval_utils import load_schemas_to_dict
from utils import represent_schema


def parse_sql(schema, sql):
    # def extract_sql(schema, sql):
    #     def extract_table(schema, table_unit):
    #         index = table_unit[1]
    #         table_name = schema['table_names_original'][index]
    #         return table_name
    
    #     target = set()
    #     for unit in sql['from']['table_units']:
    #         if unit[0] == 'table_unit':
    #             target.add(extract_table(schema, unit))
    #         elif unit[0] == 'sql':
    #             target |= extract_sql(schema, unit[1])
    #     for k in ('intersect', 'except', 'union'):
    #         if sql[k]:
    #             target |= extract_sql(schema, sql[k])
    #     return target
    
    # target = extract_sql(schema, sql)
    sql = str(sql)
    sql = re.sub('\\s+', '', sql)
    matches = re.findall('\'table_unit\',\d+\]', sql)
    return set([schema['table_names_original'][int(m[m.index(',') + 1 : -1])] for m in matches])


def prepare_spider_data(schema_file, data_file, output_file):
    schemas = load_schemas_to_dict(schema_file)
    train = json.load(open(data_file, 'r'))
    
    data = []
    for db_id in train:
        for item in train[db_id]:
            targets = parse_sql(schemas[db_id], item['sql'])
            
            inputs = []
            labels = []
            tables, primary, foreign = represent_schema(schemas[db_id])
            for table_id in tables:
                text_list = [table_id]
                # for column, ty in tables[table_id]:
                #     text = column
                #     # if table_id in primary and column == primary[table_id]:
                #     #     text += ' (primary_key)'
                #     # elif table_id in foreign and column in foreign[table_id]:
                #     #     text += ' (foreign_key {}.{})'.format(foreign[table_id][column][0], foreign[table_id][column][1])
                #     text_list.append(text)
                label = 1 if table_id in targets else 0
                inputs.append('; '.join(text_list))
                labels.append(label)
            data.append({
                'db_id': db_id,
                'question': item['question'],
                'tables': inputs,
                'labels': labels
            })
    json.dump(data, open(output_file, 'w'), indent=2)


def prepare_multiturn_data(schema_file, data_file, output_file):
    schemas = load_schemas_to_dict(schema_file)
    train = json.load(open(data_file, 'r'))
    
    data = []
    for db_id in train:
        tables, primary, foreign = represent_schema(schemas[db_id])
        for items in train[db_id]:
            prefix = ''
            for item in items:
                prefix += item['question']
                targets = parse_sql(schemas[db_id], item['sql'])
                
                inputs = []
                labels = []
                # tables, primary, foreign = represent_schema(schemas[db_id])
                for table_id in tables:
                    text_list = [table_id]
                    # for column, ty in tables[table_id]:
                    #     text = column
                    #     # if table_id in primary and column == primary[table_id]:
                    #     #     text += ' (primary_key)'
                    #     # elif table_id in foreign and column in foreign[table_id]:
                    #     #     text += ' (foreign_key {}.{})'.format(foreign[table_id][column][0], foreign[table_id][column][1])
                    #     text_list.append(text)
                    label = 1 if table_id in targets else 0
                    inputs.append('; '.join(text_list))
                    labels.append(label)
                data.append({
                    'db_id': db_id,
                    'question': item['question'],
                    'tables': inputs,
                    'labels': labels
                })
                
                prefix += '\n' + item['query'] + '\n'
    json.dump(data, open(output_file, 'w'), indent=2)           


if __name__ == '__main__':
    prepare_spider_data(
        PREDICT_DIR + '../dataset/spider_tables.json',
        PREDICT_DIR + '../dataset/spider_dev.json',
        PREDICT_DIR + '../rlhf_result/predict_table_spider_test.json'
    )
    # prepare_multiturn_data(
    #     PREDICT_DIR + '../dataset/cosql_tables.json', 
    #     PREDICT_DIR + '../dataset/cosql_dev.json',
    #     PREDICT_DIR + '../rlhf_result/predict_col_cosql_test.json'
    # )
