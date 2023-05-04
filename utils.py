import json
import re
import random
import sqlparse


def load_cosql():
    train = json.load(open('/local2/jiang719/databases/cosql/sql_state_tracking/cosql_train.json', 'r'))
    dev = json.load(open('/local2/jiang719/databases/cosql/sql_state_tracking/cosql_dev.json', 'r'))
    
    train_data, dev_data = {}, {}
    for data in train:
        id = data['database_id']
        if id not in train_data:
            train_data[id] = []
        interaction = [
            {
                'question': d['utterance'], 
                'query': sqlparse.format(
                    re.sub('\\s+', ' ', d['query']), keyword_case='upper', identifier_case='lower'
                ),
                # 'sql': d['sql']
            }
            for d in data['interaction']# + [data['final']]
        ]
        train_data[id].append(interaction)
    for data in dev:
        id = data['database_id']
        if id not in dev_data:
            dev_data[id] = []
        interaction = [
            {
                'question': d['utterance'], 
                'query': sqlparse.format(
                    re.sub('\\s+', ' ', d['query']), keyword_case='upper', identifier_case='lower'
                ),
                # 'sql': d['sql']
            }
            for d in data['interaction']# + [data['final']]
        ]
        dev_data[id].append(interaction)
    
    json.dump(train_data, open('dataset/cosql_train.json', 'w'), indent=2)
    json.dump(dev_data, open('dataset/cosql_dev.json', 'w'), indent=2)
    return train_data, dev_data


def load_sparc():
    train = json.load(open('/local2/jiang719/databases/sparc/train.json', 'r'))
    dev = json.load(open('/local2/jiang719/databases/sparc/dev.json', 'r'))
    
    train_data, dev_data = {}, {}
    for data in train:
        id = data['database_id']
        if id not in train_data:
            train_data[id] = []
        interaction = [
            {
                'question': d['utterance'], 
                'query': sqlparse.format(
                    re.sub('\\s+', ' ', d['query']), keyword_case='upper', identifier_case='lower'
                ),
                # 'sql': d['sql']
            }
            for d in data['interaction']
        ]
        train_data[id].append(interaction)
    for data in dev:
        id = data['database_id']
        if id not in dev_data:
            dev_data[id] = []
        interaction = [
            {
                'question': d['utterance'], 
                'query': sqlparse.format(
                    re.sub('\\s+', ' ', d['query']), keyword_case='upper', identifier_case='lower'
                ),
                # 'sql': d['sql']
            }
            for d in data['interaction']
        ]
        dev_data[id].append(interaction)
    
    json.dump(train_data, open('dataset/sparc_train.json', 'w'), indent=2)
    json.dump(dev_data, open('dataset/sparc_dev.json', 'w'), indent=2)
    return train_data, dev_data


def load_spider():
    train = json.load(open('/local2/jiang719/databases/spider/train_spider.json', 'r')) + \
        json.load(open('/local2/jiang719/databases/spider/train_others.json', 'r'))
    dev = json.load(open('/local2/jiang719/databases/spider/dev.json', 'r'))
    
    train_data, dev_data = {}, {}
    for data in train:
        id = data['db_id']
        if id not in train_data:
            train_data[id] = []
        train_data[id].append({
            'question': data['question'],
            'query': data['query']
            # 'sql': data['sql']
        })
    for data in dev:
        id = data['db_id']
        if id not in dev_data:
            dev_data[id] = []
        if data['question'][-2] == ' ':
            data['question'] = data['question'][:-2] + data['question'][-1]
        dev_data[id].append({
            'question': data['question'],
            'query': data['query']
            # 'sql': data['sql']
        })
    json.dump(train_data, open('dataset/spider_train.json', 'w'), indent=2)
    json.dump(dev_data, open('dataset/spider_dev.json', 'w'), indent=2)
    return train_data, dev_data


def represent_schema(database):
    tables = {}
    for (index, col), ty in zip(database['column_names_original'], database['column_types']):
        if index == -1:
            continue
        table = database['table_names_original'][index]
        if table not in tables:
            tables[table] = []
        tables[table].append((col, ty))
    
    primary = {}
    for index, table in zip(database['primary_keys'], database['table_names_original']):
        key = database['column_names_original'][index][1]
        primary[table] = key
    foreign = {}
    for start, end in database['foreign_keys']:
        index, start = database['column_names_original'][start]
        start_table = database['table_names_original'][index]
        index, end = database['column_names_original'][end]
        end_table = database['table_names_original'][index]
        if start_table not in foreign:
            foreign[start_table] = {}
        foreign[start_table][start] = (end_table, end)
        
    return tables, primary, foreign


def represent_schema_text(database, columnset={}, shuffle=False):
    tables, primary, foreign = represent_schema(database)
    text = []
    total = sum([len(table) for table in tables.values()])
    for table_id in tables:
        # table_text = table_id + ': '
        table_text = table_id + ' : '
        columns = []
        for column, ty in tables[table_id]:
            # if (table_id + '.' + column).lower() not in columnset:
            #     continue
            tmp = column
            if table_id in primary and column == primary[table_id]:
                # tmp += ' (primary_key)'
                tmp += ' ( primary_key )'
            # elif table_id in foreign and column in foreign[table_id] and \
            #     (foreign[table_id][column][0] + '.' + foreign[table_id][column][1]).lower() in columnset:
            elif table_id in foreign and column in foreign[table_id]:
                # tmp += ' (foreign_key {}.{})'.format(foreign[table_id][column][0], foreign[table_id][column][1])
                tmp += ' ( foreign_key {} . {} )'.format(foreign[table_id][column][0], foreign[table_id][column][1])
            columns.append(tmp)
        if columns != []:
            if shuffle:
                random.shuffle(columns)
            # text.append(table_text + ', '.join(columns) + '; ')
            text.append(table_text + ' , '.join(columns) + ' ; ')
    if shuffle:
        random.shuffle(text)
    return ''.join(text).lower()


def represent_schema_code(database, columnset={}, shuffle=False):
    tables, primary, foreign = represent_schema(database)
    text = []
    for table_id in tables:
        table_text = 'CREATE TABLE ' + table_id + ' ('
        columns = []
        for column, ty in tables[table_id]:
            if (table_id + '.' + column).lower() not in columnset:
                continue
            tmp = column + ' ' + ty.upper()
            # if table_id in primary and column == primary[table_id]:
            #     tmp += ' PRIMARY KEY'
            tmp += ','
            columns.append(tmp)
        if columns == []:
            continue
        if shuffle:
            random.shuffle(columns)
        # if table_id in foreign:
        #     for c in foreign[table_id]:
        #         columns.append(
        #             'FOREIGN KEY ({}) REFERENCES {} ({}),'.format(c, foreign[table_id][c][0], foreign[table_id][c][1])
        #         )
        columns[-1] = columns[-1][:-1]
        text.append(table_text + ' '.join(columns) + ');')
    if shuffle:
        random.shuffle(text)
    return ' '.join(text)


if __name__ == '__main__':
    load_spider()
    # load_sparc()
    # load_cosql()
    # data = json.load(open('dataset/spider_tables.json', 'r'))
    # data = {
    #     d['db_id']: d for d in data
    # }
    # text = represent_schema_text(data['student_assessment'])
    # print(text)
