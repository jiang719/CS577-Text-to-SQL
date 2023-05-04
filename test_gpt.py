import json
import matplotlib.pyplot as plt
import openai
import tiktoken
from utils import represent_schema_text


openai.api_key = None
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def database_text_length():
    length = {
        '(0,128]': 0, '(128,256]': 0, '(256,512]': 0,
        '(512,1024]': 0, '(1024,inf)':0
    }
    file = 'dataset/spider_tables.json'
    data = json.load(open(file, 'r'))
    for db in data:
        text = represent_schema_text(db)
        if len(text) > 1e5:
            l = '(1024,inf)'
        else:
            tokens = enc.encode(text.strip())
            l = min(2048, len(tokens))
            if l <= 128:
                l = '(0,128]'
            elif l <= 256:
                l = '(128,256]'
            elif l <= 512:
                l = '(256,512]'
            elif l <= 1024:
                l = '(512,1024]'
            else:
                l = '(1024,inf)'
        length[l] += 1
        
    plt.bar(length.keys(), length.values())
    plt.xticks([0, 1, 2, 3, 4], length.keys(), rotation=30)
    plt.xlabel('#Tokens in Database Representation', fontsize=14)
    plt.ylabel('#Database', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/representation_length.png', dpi=300, bbox_inches='tight')


def database_table_num():
    num = {
        '(0,2]': 0, '(2,4]': 0, '(4,8]': 0,
        '(8,16]': 0, '(16,32]': 0
    }
    file = 'dataset/spider_tables.json'
    schemas = json.load(open(file, 'r'))
    for schema in schemas:
        n = len(schema['table_names'])
        if n <= 2:
            n = '(0,2]'
        elif n <= 4:
            n = '(2,4]'
        elif n <= 8:
            n = '(4,8]'
        elif n <= 16:
            n = '(8,16]'
        elif n <= 32:
            n = '(16,32]'
        num[n] += 1
    plt.figure(figsize=(5, 4))
    plt.bar(num.keys(), num.values())
    plt.xticks([0, 1, 2, 3, 4], num.keys(), fontsize=18, rotation=30)
    plt.yticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'], fontsize=18)
    plt.xlabel('#Tables in Database', fontsize=20)
    plt.ylabel('#Database', fontsize=20)
    plt.tight_layout()
    plt.savefig('figures/schema_table_num.pdf', bbox_inches='tight')


def database_column_num():
    num = {
        '(0,16]': 0, '(16,32]': 0, '(32,64]': 0,
        '(64,128]': 0, '(128,inf)': 0
    }
    file = 'dataset/spider_tables.json'
    schemas = json.load(open(file, 'r'))
    for schema in schemas:
        n = len(schema['column_names_original']) - 1
        if n <= 16:
            n = '(0,16]'
        elif n <= 32:
            n = '(16,32]'
        elif n <= 64:
            n = '(32,64]'
        elif n <= 128:
            n = '(64,128]'
        elif n <= 512:
            n = '(128,inf)'
        num[n] += 1
    plt.figure(figsize=(5, 4))
    plt.bar(num.keys(), num.values())
    plt.xticks([0, 1, 2, 3, 4], num.keys(), fontsize=18, rotation=30)
    plt.yticks([0, 20, 40, 60, 80], ['0', '20', '40', '60', '80'], fontsize=18)
    plt.xlabel('#Columns in Database', fontsize=20)
    plt.ylabel('#Database', fontsize=20)
    plt.tight_layout()
    plt.savefig('figures/schema_column_num.pdf', bbox_inches='tight')


def database_table_length():
    length = {
        '(0,128]': 0, '(128,256]': 0, '(256,512]': 0,
        '(512,1024]': 0, '(1024,2048]': 0, '(2048,4096]': 0, '(4096,inf)':0
    }
    file = 'dataset/schema.json'
    schemas = json.load(open(file, 'r'))
    for db, schema in schemas.items():
        for table in schema.split('CREATE')[1:]:
            if len(table) > 1e5:
                l = '(4096,inf)'
            else:
                tokens = enc.encode(table.strip())
                l = min(4096, len(tokens))
                if l <= 128:
                    l = '(0,128]'
                elif l <= 256:
                    l = '(128,256]'
                elif l <= 512:
                    l = '(256,512]'
                elif l <= 1024:
                    l = '(512,1024]'
                elif l <= 2048:
                    l = '(1024,2048]'
                elif l <= 4096:
                    l = '(2048,4096]'
                else:
                    l = '(4096,inf)'
            length[l] += 1
        
    plt.bar(length.keys(), length.values())
    plt.xticks([0, 1, 2, 3, 4, 5, 6], length.keys(), rotation=30)
    plt.xlabel('#Tokens in Table Schema', fontsize=14)
    plt.ylabel('#Tables', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/table_length.png', dpi=300, bbox_inches='tight')


def database_length():
    length = {
        '(0,128]': 0, '(128,256]': 0, '(256,512]': 0,
        '(512,1024]': 0, '(1024,2048]': 0, '(2048,4096]': 0, '(4096,inf)':0
    }
    file = '/local2/jiang719/databases/spider/dataset/schema.json'
    schemas = json.load(open(file, 'r'))
    for db, schema in schemas.items():
        if len(schema) > 1e5:
            l = '(4096,inf)'
        else:
            tokens = enc.encode(schema.strip())
            l = min(4096, len(tokens))
            if l <= 128:
                l = '(0,128]'
            elif l <= 256:
                l = '(128,256]'
            elif l <= 512:
                l = '(256,512]'
            elif l <= 1024:
                l = '(512,1024]'
            elif l <= 2048:
                l = '(1024,2048]'
            elif l <= 4096:
                l = '(2048,4096]'
            else:
                l = '(4096,inf)'
        length[l] += 1
        
    plt.bar(length.keys(), length.values())
    plt.xticks([0, 1, 2, 3, 4, 5, 6], length.keys(), rotation=30)
    plt.xlabel('#Tokens in Database Schema', fontsize=14)
    plt.ylabel('#Database', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/schema_length.png', dpi=300, bbox_inches='tight')


def generate_chatgpt(prompt, max_tokens=128):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stop=['###']
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    # prompt = 'Given a database and its schema, convert the natural language question to sql query.\n'
    # prompt += 'CREATE TABLE \"continents\" ( \n\t\"ContId\" INTEGER PRIMARY KEY, \n\t\"Continent\" TEXT \n);\n\nCREATE TABLE \"countries\" (\n\t\"CountryId\" INTEGER PRIMARY KEY, \n\t\"CountryName\" TEXT, \n\t\"Continent\" INTEGER,\n\tFOREIGN KEY (Continent) REFERENCES continents(ContId)\n);\n\n\nCREATE TABLE \"car_makers\" ( \n\t\"Id\" INTEGER PRIMARY KEY, \n\t\"Maker\" TEXT, \n\t\"FullName\" TEXT, \n\t\"Country\" TEXT,\n\tFOREIGN KEY (Country) REFERENCES countries(CountryId)\n);\n\n\nCREATE TABLE \"model_list\" ( \n\t\"ModelId\" INTEGER PRIMARY KEY, \n\t\"Maker\" INTEGER, \n\t\"Model\" TEXT UNIQUE,\n\tFOREIGN KEY (Maker) REFERENCES car_makers (Id)\n\n);\n\n\n\nCREATE TABLE \"car_names\" ( \n\t\"MakeId\" INTEGER PRIMARY KEY, \n\t\"Model\" TEXT, \n\t\"Make\" TEXT,\n\tFOREIGN KEY (Model) REFERENCES model_list (Model)\n);\n\nCREATE TABLE \"cars_data\" (\n\t\"Id\" INTEGER PRIMARY KEY, \n\t\"MPG\" TEXT, \n\t\"Cylinders\" INTEGER, \n\t\"Edispl\" REAL, \n\t\"Horsepower\" TEXT, \n\t\"Weight\" INTEGER, \n\t\"Accelerate\" REAL, \n\t\"Year\" INTEGER,\n\tFOREIGN KEY (Id) REFERENCES car_names (MakeId)\n);\n'
    # prompt += 'question: What are all the makers and models?\n'
    # prompt += 'sql: '
    
    # print(generate_chatgpt(prompt))
    # database_length()
    # database_table_length()
    database_table_num()
    # database_column_num()
    # database_text_length()
