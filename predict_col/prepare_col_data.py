import copy
import json
import os
import sys
import numpy as np

PREDICT_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(PREDICT_DIR + '../')
from eval_utils import load_schemas_to_dict
from utils import represent_schema


def parse_sql(schema, sql):
    def extract_sql(schema, sql):
        def extract_col(schema, col_unit):
            col_index = col_unit[1]
            table_index, col_name = schema['column_names_original'][col_index]
            table_name = schema['table_names_original'][table_index]
            if col_name == '*':
                return None
            return table_name + '.' + col_name
    
        target = set()
        if sql['groupBy']:
            for col_unit in sql['groupBy']:
                target.add(extract_col(schema, col_unit))
        if sql['select'] and sql['select'][1]:
            for item in sql['select'][1]:
                for col_unit in item[1][1:]:
                    if col_unit:
                        target.add(extract_col(schema, col_unit))
        if sql['orderBy'] and sql['orderBy'][1]:
            for item in sql['orderBy'][1]:
                for col_unit in item[1:]:
                    if col_unit:
                        target.add(extract_col(schema, col_unit))
        if sql['from']['table_units']:
            for unit in sql['from']['table_units']:
                if unit[0] == 'sql':
                    target |= extract_sql(schema, unit[1])
        if sql['from']['conds']:
            for i in range(0, len(sql['from']['conds']), 2):
                cond = sql['from']['conds'][i]
                val_unit = cond[2]
                for col_unit in val_unit[1:]:
                    if col_unit:
                        target.add(extract_col(schema, col_unit))
                if cond[3] and type(cond[3]) == list and len(cond[3]) == 3:
                    target.add(extract_col(schema, cond[3]))
                if cond[4] and type(cond[4]) == list and len(cond[4]) == 3:
                    target.add(extract_col(schema, cond[4]))
        if sql['where']:
            for i in range(0, len(sql['where']), 2):
                cond = sql['where'][i]
                val_unit = cond[2]
                for col_unit in val_unit[1:]:
                    if col_unit:
                        target.add(extract_col(schema, col_unit))
        if sql['having']:
            for i in range(0, len(sql['having']), 2):
                cond = sql['having'][i]
                val_unit = cond[2]
                for col_unit in val_unit[1:]:
                    if col_unit:
                        target.add(extract_col(schema, col_unit))
        for k in ('intersect', 'except', 'union'):
            if sql[k]:
                target |= extract_sql(schema, sql[k])
        return target
    
    target = extract_sql(schema, sql)
    if None in target:
        target.remove(None)
    return target


def prepare_spider_data(schema_file, data_file, output_file):
    schemas = load_schemas_to_dict(schema_file)
    train = json.load(open(data_file, 'r'))
    
    data = []
    for db_id in train:
        cnt = 0
        for item in train[db_id]:
            tables, primary, foreign = represent_schema(schemas[db_id])
            targets = parse_sql(schemas[db_id], item['sql'])
            if len(targets) == 0:
                table_id = item['query'].strip().split()[-1]
                if table_id[-1] == ';':
                    table_id = table_id[: -1]
                for k in tables.keys():
                    if k.lower() == table_id.lower():
                        table_id = k
                        break
                text = table_id + '.' + tables[table_id][0][0]
                targets.add(text)
                
            columns = []
            labels = []
            for table_id in tables:
                for column, ty in tables[table_id]:
                    text = table_id + '.' + column
                    label = 1 if text in targets else 0
                    # if table_id in primary and column == primary[table_id]:
                    #     text += ' (primary)'
                    # elif table_id in foreign and column in foreign[table_id]:
                    #     text += ' (foreign {}.{})'.format(foreign[table_id][column][0], foreign[table_id][column][1])
                    columns.append(text)
                    labels.append(label)
            adj = np.zeros((len(columns), len(columns)))
            index = 0
            raw_colomns = [table_id + '.' + column for table_id in tables for column, ty in tables[table_id]]
            # if len(raw_colomns) > 128:
            #     print(db_id)
            #     continue
            for table_id in tables:
                adj[index: index + len(tables[table_id]), index: index + len(tables[table_id])] = \
                    np.ones((len(tables[table_id]), len(tables[table_id]))) - np.eye(len(tables[table_id]))
                index += len(tables[table_id])
                for column, ty in tables[table_id]:
                    if table_id in foreign and column in foreign[table_id]:
                        foreign_col = foreign[table_id][column][0] + '.' + foreign[table_id][column][1]
                        start_index = raw_colomns.index(foreign_col)
                        end_index = raw_colomns.index(table_id + '.' + column)
                        adj[end_index][start_index] = 2
                        adj[start_index][end_index] = 3
            
            data.append({
                'db_id': db_id,
                'index': cnt,
                'question': item['question'],
                'query': item['query'],
                'columns': columns,
                'labels': labels,
                'adj': adj.astype(int).tolist()
            })
            cnt += 1
    json.dump(data, open(output_file, 'w'), indent=2)


def prepare_multiturn_data(schema_file, data_file, output_file):
    schemas = load_schemas_to_dict(schema_file)
    train = json.load(open(data_file, 'r'))
    
    data = []
    for db_id in train:
        cnt = [0, 0]
        tables, primary, foreign = represent_schema(schemas[db_id])
        for items in train[db_id]:
            prefix = ''
            cnt[1] = 0
            for item in items:
                prefix += item['question']
                targets = parse_sql(schemas[db_id], item['sql'])
                try:
                    if len(targets) == 0:
                        table_id = item['query'].strip().split()[-1]
                        if table_id[-1] == ';':
                            table_id = table_id[: -1]
                        for k in tables.keys():
                            if k.lower() == table_id.lower():
                                table_id = k
                                break
                        text = table_id + '.' + tables[table_id][0][0]
                        targets.add(text)
                except:
                    print(item['query'])
                
                columns = []
                labels = []
                for table_id in tables:
                    for column, ty in tables[table_id]:
                        text = table_id + '.' + column
                        label = 1 if text in targets else 0
                        # if table_id in primary and column == primary[table_id]:
                        #     text += ' (primary_key)'
                        # elif table_id in foreign and column in foreign[table_id]:
                        #     text += ' (foreign_key {}.{})'.format(foreign[table_id][column][0], foreign[table_id][column][1])
                        columns.append(text)
                        labels.append(label)
                adj = np.zeros((len(columns), len(columns)))
                index = 0
                raw_colomns = [table_id + '.' + column for table_id in tables for column, ty in tables[table_id]]
                # if len(raw_colomns) > 128:
                #     print(db_id)
                #     continue
                for table_id in tables:
                    adj[index: index + len(tables[table_id]), index: index + len(tables[table_id])] = \
                        np.ones((len(tables[table_id]), len(tables[table_id]))) - np.eye(len(tables[table_id]))
                    index += len(tables[table_id])
                    for column, ty in tables[table_id]:
                        if table_id in foreign and column in foreign[table_id]:
                            foreign_col = foreign[table_id][column][0] + '.' + foreign[table_id][column][1]
                            start_index = raw_colomns.index(foreign_col)
                            end_index = raw_colomns.index(table_id + '.' + column)
                            adj[end_index][start_index] = 2
                            adj[start_index][end_index] = 3
                
                data.append({
                    'db_id': db_id,
                    'index': copy.deepcopy(cnt),
                    'question': prefix,
                    'query': item['query'],
                    'columns': columns,
                    'labels': labels,
                    'adj': adj.astype(int).tolist()
                })
                cnt[1] += 1
                prefix += '\n' + item['query'] + '\n'
            cnt[0] += 1
    json.dump(data, open(output_file, 'w'), indent=2)           


if __name__ == '__main__':
    # prepare_multiturn_data(
    #     PREDICT_DIR + '../dataset/sparc_tables.json', 
    #     PREDICT_DIR + '../dataset/sparc_train.json',
    #     PREDICT_DIR + '../predict_col_result/predict_col_sparc_train.json'
    # )
    prepare_spider_data(
        PREDICT_DIR + '../dataset/spider_tables.json',
        PREDICT_DIR + '../dataset/spider_train.json',
        PREDICT_DIR + '../predict_col_result/predict_col_spider_train.json'
    )
