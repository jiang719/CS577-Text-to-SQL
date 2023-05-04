import os
import sys
import time
import numpy as np
import torch
from dataset import Dataset
from transformers import AutoTokenizer
from transformers import Adafactor
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

RLHF_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(RLHF_DIR + '../../')

from evaluation import build_foreign_key_map_from_json, evaluate


def reinforcement_learning(training_files, epoches, beam_size, model_path, save_dir):
    device_map = {'model.embed_tokens': 0, 'model.embed_positions': 0, 'model.layer_norm': 0, 'lm_head': 0}
    device_map.update({'model.layers.' + str(i): 0 for i in range(0, 7)})
    device_map.update({'model.layers.' + str(i): 1 for i in range(7, 14)})
    device_map.update({'model.layers.' + str(i): 2 for i in range(14, 19)})
    device_map.update({'model.layers.' + str(i): 3 for i in range(19, 24)})
    device_ids = [0, 1, 2, 3]

    tokenizer = AutoTokenizer.from_pretrained('facebook/incoder-1B')
    eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')
    tokenizer.add_special_tokens({'pad_token': '<|endofmask|>'})
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, device_map=device_map)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, device_map=device_map)
    # model_ref = create_reference_model(model, num_shared_layers=12)

    ppo_config = {'batch_size': beam_size + 1, 'forward_batch_size': 2, 'learning_rate': 1e-5, 
                  'optimize_cuda_cache': True, 'ppo_epochs': 4}
    config = PPOConfig(**ppo_config)
    optimizer = Adafactor(model.parameters(), lr=config.learning_rate, scale_parameter=False, relative_step=False)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, optimizer=optimizer)

    training_dataset = Dataset(training_files[0], tokenizer, max_length=1536, shuffle=True)
    for epoch in range(epoches):
        train_rewards = []
        train_ppo_loss = []
        start_time = time.time()
        oom = 0
        for i in range(len(training_dataset)):
            data = training_dataset[i]
            input_ids = data['input_ids'].to(device_ids[0])
            output_ids = data['output_ids'].to(device_ids[0])

            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model.generate(
                        input_ids, max_new_tokens=128, num_beams=beam_size, num_return_sequences=beam_size, 
                        early_stopping=True, pad_token_id=eos_id, eos_token_id=eos_id
                    )

                max_len = max(output_ids.size(1), outputs.size(1))
                output_ids = torch.cat([
                    torch.cat([output_ids, torch.zeros(output_ids.size(0), max_len - output_ids.size(1)).fill_(eos_id).long().to(output_ids.device)], dim=1), 
                    torch.cat([outputs, torch.zeros(outputs.size(0), max_len - outputs.size(1)).fill_(eos_id).long().to(outputs.device)], dim=1)
                ], dim=0)
                output_ids = [output for output in output_ids]

                output_texts = [
                    tokenizer.decode(output[input_ids.size(1): ], skip_special_tokens=True).strip() for output in output_ids
                ]
                db_name = data['db_name']
                rewards = [
                    torch.tensor(evaluate(
                        output_texts[0], output, db_name, '/local2/jiang719/databases/spider/database/', kmaps
                    )).to(input_ids.device) for output in output_texts
                ]
                rewards = [2*r*r - 1 for r in rewards]

                stats = ppo_trainer.step([torch.clone(input_ids.squeeze(0)) for _ in range(1 + beam_size)], output_ids, rewards)
                train_rewards.append(np.mean([reward.tolist() for reward in rewards[1:]]))
                train_ppo_loss.append(stats['ppo/loss/total'])
            except:
                oom += 1
                torch.cuda.empty_cache()
            
            if i % 10 == 0:
                print('step: {}, rewards: {}, loss: {}, time: {}s, oom: {}'.format(
                    i, round(np.mean(train_rewards), 6), round(np.mean(train_ppo_loss), 6), 
                    int(time.time() - start_time), oom)
                )
                start_time = time.time()
                oom = 0
            if i % 100 == 0:
                model.save_pretrained(save_dir)


if __name__ == '__main__':
    kmaps = build_foreign_key_map_from_json(RLHF_DIR + '../../dataset/spider_tables.json')
    reinforcement_learning(
        training_files=[
            RLHF_DIR + '../../rlhf_result/finetuning_spider_train.jsonl'
        ],
        epoches=1, beam_size=9,
        model_path='/local2/jiang719/language_models/incoder/incoder-1B-sql/',
        save_dir='/local2/jiang719/language_models/incoder/incoder-1B-sql-rlhf/'
    )
