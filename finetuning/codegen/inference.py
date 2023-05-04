import re
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, SQLLogitsProcessor, LogitsProcessorList
import os
import sys
from nltk.tokenize import word_tokenize

CODEGEN_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(CODEGEN_DIR + '../../')
from eval_utils import load_schemas_to_dict, represent_schema_text
from evaluation import build_foreign_key_map_from_json, parse

keywords = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except') + \
    ('join', 'on', 'as') + ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists') + \
    ('-', '+', "*", '/') + ('max', 'min', 'count', 'sum', 'avg') + ('and', 'or') + ('intersect', 'union', 'except') + \
    ('desc', 'asc') + ('by', 'having', 'distinct', 't1', 't2', 't3', 't4', 't5', 't6', 't7', '.') + \
    ('1', '"', ' ', '(', ')', ';')
keywords = set(keywords)


def process_question(question):
    question = word_tokenize(question)
    question = ' '.join(question)
    question = question.replace('(', ' ( ')
    question = question.replace(')', ' ) ')
    question = question.replace('.', ' . ')
    question = question.replace('"', ' " ')
    question = question.replace("'", ' " ')
    return re.sub('\\s+', ' ', question).strip()


def process_query(query):
    query = query.replace('(', ' ( ')
    query = query.replace(')', ' ) ')
    query = query.replace('"', ' " ')
    query = query.replace("'", ' " ')
    query = query.replace('.', ' . ')
    query = query.replace(',', ' , ')
    query = query.replace('>', ' > ')
    query = query.replace('<', ' < ')
    query = query.replace(';', '')
    query = re.sub('\\s+', ' ', query)
    
    lst = query.split('"')
    query = ''
    for i in range(0, len(lst), 2):
        if i < len(lst) - 1:
            query += lst[i].lower() + '"' + lst[i+1] + '"'
        else:
            query += lst[i].lower()
    query = re.sub('\\s+', ' ', query)
    return query.strip()


def postprocess_query(query):
    query = query.replace(' . ', '.')
    query = query.replace('. ', '.')
    query = query.replace('_ ', '_')
    query = query.replace(' _', '_')
    query = query.replace('( ', '(')
    query = query.replace(' )', ')')
    query = query.replace('> =', '>=')
    query = query.replace('< =', '<=')
    query = query.replace('! =', '!=')
    lst = query.split('"')
    query = ''
    for i in range(0, len(lst), 2):
        if i < len(lst) - 1:
            query += lst[i] + '"' + lst[i+1].strip() + '"'
        else:
            query += lst[i]
    
    query = query.replace('"english"', '"English"')
    query = query.replace('"france"', '"France"')
    # query = query.replace('friend_id', 'student_id')

    return re.sub('\\s+', ' ', query).strip()


def process_vocabulary(schema, tokenizer):
    voc = set(schema.lower().strip().split())
    voc |= set(keywords)
    dfa = {}
    first_ids = set()
    for word in voc:
        ids = tokenizer.encode('select ' + word)[1:]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        prefix = ''
        first_ids.add(ids[0])
        for id, token in zip(ids, tokens):
            if prefix not in dfa:
                dfa[prefix] = set()
            dfa[prefix].add(id)
            prefix += token
    for word in voc:
        if 'Ġ' + word not in dfa:
            dfa['Ġ' + word] = set()
        dfa['Ġ' + word] |= first_ids
        dfa['Ġ' + word].add(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
    return dfa


def inference_sparc(test_file, output_file, vocabulary_file, model_file):
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
    model = AutoModelForCausalLM.from_pretrained(model_file, device_map=device_map).half()
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    
    data = json.load(open(test_file, 'r'))
    result = json.load(open(CODEGEN_DIR + '../../finetuning_rlhf_result/sparc_dev_result_codegen_2b.json'))
    # if os.path.exists(output_file):
    #     result = json.load(open(output_file, 'r'))
    # else:
    #     result = {}
    schemas = load_schemas_to_dict(CODEGEN_DIR + '../../dataset/sparc_tables.json')
    cnt = 0
    oom = 0
    for db_id in data:
        # if db_id in result:
        #     if len(result[db_id]) == len(data[db_id]):
        #         continue
        #     else:
        #         pass
        # else:
        #     result[db_id] = []
        # for iteraction in data[db_id][len(result[db_id]): ]:
        for index, iteraction in enumerate(data[db_id]):
            parsed = True
            for iter in result[db_id][index]:
                if len(iter['query']) > 1:
                    continue
                if parse(iter['query'][0], db_id, '/local2/jiang719/databases/sparc/database/'):
                    continue
                parsed = False
                break
            if parsed:
                continue
            print(db_id, index)
            iter_result = []
            inputs ='Schema: ' + represent_schema_text(schemas[db_id])
            for item in iteraction:
                inputs += '\nQuestion: ' + process_question(item['question']).strip() + '\nQuery: '

                inputs_tensor = tokenizer(inputs, return_tensors="pt").input_ids
                inputs_tensor = inputs_tensor.to(device_ids[0])
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                
                dfa = process_vocabulary(item['schema'].strip(), tokenizer)
                logits_processor = LogitsProcessorList(
                    [SQLLogitsProcessor(dfa, tokenizer, inputs_tensor.size(1))]
                )
                try:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model.generate(
                            inputs_tensor, max_new_tokens=160, num_beams=16, num_return_sequences=16, 
                            logits_processor=logits_processor,
                            early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                        )
                except:
                    torch.cuda.empty_cache()
                    try:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model.generate(
                                inputs_tensor, max_new_tokens=160, num_beams=8, num_return_sequences=8, 
                                logits_processor=logits_processor,
                                early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                            )
                    except:
                        torch.cuda.empty_cache()
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model.generate(
                                inputs_tensor, max_new_tokens=128, num_beams=4, num_return_sequences=4, 
                                logits_processor=logits_processor,
                                early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                            )
                output_texts = []
                valid_output_texts = []
                first_valid_index = -1
                for i, output in enumerate(outputs):
                    output = output[inputs_tensor.size(1): ]
                    output = tokenizer.decode(output, skip_special_tokens=True)
                    output = postprocess_query(output)
                    if output == '':
                        continue
                    if not output.startswith('select'):
                        output = 'select ' + output
                    output_texts.append(output)
                    if parse(output, db_id, '/local2/jiang719/databases/sparc/database/'):
                        valid_output_texts.append(output)
                        if first_valid_index == -1:
                            first_valid_index = i
                iter_result.append({
                    'question': item['question'],
                    'query': valid_output_texts[:10] if valid_output_texts else [output_texts[0]]
                })
                
                inputs += process_query(valid_output_texts[0]) if valid_output_texts else process_query(output_texts[0])
                cnt += 1
                print(cnt, first_valid_index, valid_output_texts != [], oom)
                
            result[db_id][index] = iter_result
            # result[db_id].append(iter_result)
            json.dump(result, open(output_file, 'w'), indent=2)


def inference_cosql(test_file, output_file, vocabulary_file, model_file):
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
    model = AutoModelForCausalLM.from_pretrained(model_file, device_map=device_map).half()
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    
    data = json.load(open(test_file, 'r'))
    result = json.load(open(CODEGEN_DIR + '../../finetuning_rlhf_result/cosql_dev_result_codegen_2b.json'))
    # if os.path.exists(output_file):
    #     result = json.load(open(output_file, 'r'))
    # else:
    #     result = {}
    schemas = load_schemas_to_dict(CODEGEN_DIR + '../../dataset/cosql_tables.json')
    cnt = 0
    oom = 0
    for db_id in data:
        # if db_id in result:
        #     if len(result[db_id]) == len(data[db_id]):
        #         continue
        #     else:
        #         pass
        # else:
        #     result[db_id] = []
        # for iteraction in data[db_id][len(result[db_id]): ]:
        for index, iteraction in enumerate(data[db_id]):
            parsed = True
            for iter in result[db_id][index]:
                if len(iter['query']) > 1:
                    continue
                if parse(iter['query'][0], db_id, '/local2/jiang719/databases/sparc/database/'):
                    continue
                parsed = False
                break
            if parsed:
                continue
            print(db_id, index)
            iter_result = []
            inputs ='Schema: ' + represent_schema_text(schemas[db_id])
            for item in iteraction:
                inputs += '\nQuestion: ' + process_question(item['question']) + '\nQuery: '

                inputs_tensor = tokenizer(inputs, return_tensors="pt").input_ids
                inputs_tensor = inputs_tensor.to(device_ids[0])
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                
                dfa = process_vocabulary(item['schema'].strip(), tokenizer)
                logits_processor = LogitsProcessorList(
                    [SQLLogitsProcessor(dfa, tokenizer, inputs_tensor.size(1))]
                )
                try:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model.generate(
                            inputs_tensor, max_new_tokens=160, num_beams=12, num_return_sequences=12, 
                            logits_processor=logits_processor,
                            early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                        )
                except:
                    torch.cuda.empty_cache()
                    try:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model.generate(
                                inputs_tensor, max_new_tokens=160, num_beams=6, num_return_sequences=6, 
                                logits_processor=logits_processor,
                                early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                            )
                    except:
                        torch.cuda.empty_cache()
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model.generate(
                                inputs_tensor, max_new_tokens=128, num_beams=3, num_return_sequences=3, 
                                logits_processor=logits_processor,
                                early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                            )
                output_texts = []
                valid_output_texts = []
                first_valid_index = -1
                for i, output in enumerate(outputs):
                    output = output[inputs_tensor.size(1): ]
                    output = tokenizer.decode(output, skip_special_tokens=True)
                    output = postprocess_query(output)
                    if output == '':
                        continue
                    if not output.startswith('select'):
                        output = 'select ' + output
                    output_texts.append(output)
                    if parse(output, db_id, '/local2/jiang719/databases/sparc/database/'):
                        valid_output_texts.append(output)
                        if first_valid_index == -1:
                            first_valid_index = i
                iter_result.append({
                    'question': item['question'],
                    'query': valid_output_texts[:10] if valid_output_texts else [output_texts[0]]
                })
                
                inputs += process_query(valid_output_texts[0]) if valid_output_texts else process_query(output_texts[0])
                cnt += 1
                print(cnt, first_valid_index, valid_output_texts != [], oom)
                
            result[db_id][index] = iter_result
            # result[db_id].append(iter_result)
            json.dump(result, open(output_file, 'w'), indent=2)


def inference_spider(test_file, output_file, vocabulary_file, model_file):
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
    model = AutoModelForCausalLM.from_pretrained(model_file, device_map=device_map)# .half()
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    
    data = json.load(open(test_file, 'r'))
    result = json.load(open(CODEGEN_DIR + '../../finetuning_rlhf_result/spider_dev_col_result_codegen_2b.json'))
    # if os.path.exists(output_file):
    #     result = json.load(open(output_file, 'r'))
    # else:
    #     result = {}
    schemas = load_schemas_to_dict(CODEGEN_DIR + '../../dataset/spider_tables.json')
    cnt = 0
    for db_id in data:
        # if db_id in result:
        #     continue
        # result[db_id] = []
        for index, item in enumerate(data[db_id]):
            if len(result[db_id][index]['query']) > 1:
                continue
            if parse(result[db_id][index]['query'][0], db_id, '/local2/jiang719/databases/spider/database/'):
                continue
            # print(process_question(item['question']))
            schema = represent_schema_text(schemas[db_id])
            inputs = 'Schema: ' + schema.strip() + '\nQuestion: ' + process_question(item['question']).strip() + '\nQuery: select'
            inputs = tokenizer(inputs, return_tensors="pt").input_ids
            inputs = inputs.to(device_ids[0])
            eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
            
            dfa = process_vocabulary(item['schema'].strip(), tokenizer)
            logits_processor = LogitsProcessorList(
                [SQLLogitsProcessor(dfa, tokenizer, inputs.size(1))]
            )
            try:
                torch.cuda.empty_cache()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model.generate(
                        inputs, max_new_tokens=192, num_beams=16, num_return_sequences=16, 
                        logits_processor=logits_processor, 
                        early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                    )
            except:
                torch.cuda.empty_cache()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model.generate(
                        inputs, max_new_tokens=192, num_beams=8, num_return_sequences=8, 
                        logits_processor=logits_processor, 
                        early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                    )
            output_texts = []
            valid_output_texts = []
            first_valid_index = -1
            for i, output in enumerate(outputs):
                output = output[inputs.size(1): ]
                output = tokenizer.decode(output, skip_special_tokens=True)
                output = postprocess_query(output)
                if output == '':
                    continue
                if not output.startswith('select'):
                    output = 'select ' + output
                output_texts.append(output)
                if parse(output, db_id, '/local2/jiang719/databases/spider/database/'):
                    valid_output_texts.append(output)
                    if first_valid_index == -1:
                        first_valid_index = i

            result[db_id][index]['query'] = valid_output_texts[:10] if valid_output_texts else [output_texts[0]]
            # result[db_id].append({
            #     'question': item['question'],
            #     'query': valid_output_texts[:10] if valid_output_texts else [output_texts[0]]
            # })
            cnt += 1
            print(cnt, first_valid_index, valid_output_texts != [])
            json.dump(result, open(output_file, 'w'), indent=2)


if __name__ == '__main__':
    device_map = {'transformer.wte': 0, 'transformer.ln_f': 3, 'lm_head': 0}
    # device_map.update({'transformer.h.' + str(i): 0 for i in range(0, 2)})
    device_map.update({'transformer.h.' + str(i): 1 for i in range(0, 11)})
    device_map.update({'transformer.h.' + str(i): 2 for i in range(11, 22)})
    device_map.update({'transformer.h.' + str(i): 3 for i in range(22, 32)})
    device_ids = [0, 1, 2, 3]

    inference_sparc(
        test_file=CODEGEN_DIR + '../../dataset/sparc_dev.json',
        output_file=CODEGEN_DIR + '../../finetuning_rlhf_result/sparc_dev_col_result_codegen_2b.json', 
        vocabulary_file='Salesforce/codegen-2B-mono',
        model_file='/local2/jiang719/language_models/codegen/codegen-2B-sql-sparc/'
    )

    inference_cosql(
        test_file=CODEGEN_DIR + '../../dataset/cosql_dev.json', 
        output_file=CODEGEN_DIR + '../../finetuning_rlhf_result/cosql_dev_col_result_codegen_2b.json', 
        vocabulary_file='Salesforce/codegen-2B-mono', 
        model_file='/local2/jiang719/language_models/codegen/codegen-2B-sql-cosql/'
    )
