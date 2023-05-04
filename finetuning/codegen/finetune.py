import time
import torch
import torch.nn as nn
from dataset import Dataset, custom_collate
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup, Adafactor


def validation_step(model, validation_loader, save_dir):
    print('start validation')
    torch.cuda.empty_cache()
    validation_loss = []
    model.eval()
    oom = 0
    with torch.no_grad():
        try:
            for i, data in enumerate(validation_loader):
                data = {
                    'input_ids': data['input_ids'].to(device_ids[0]),
                    'labels': data['labels'].to(device_ids[0]),
                    'attention_mask': data['attention_mask'].to(device_ids[0])
                }
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                loss = output.loss
                validation_loss.append(loss.mean().item())
        except Exception as e:
            torch.cuda.empty_cache()
            oom += 1
            pass
    model.save_pretrained(save_dir)
    print('checkpoint saved')
    print('validation loss:', round(sum(validation_loss) / len(validation_loss), 4), ', oom:', oom)
    model.train()


def finetune(training_files, validation_files, epochs, batch_size, save_dir):
    scaler = torch.cuda.amp.GradScaler()
    tokenizer = AutoTokenizer.from_pretrained(vocabulary_file)
    model = AutoModelForCausalLM.from_pretrained(pretrained_file, device_map=device_map)# .half()
    print('model parameters:', sum(param.numel() for param in model.parameters()))

    training_dataset = Dataset(training_files[0], tokenizer, max_length=1536, shuffle=True)
    for file in training_files[1:]:
        training_dataset.merge(
            Dataset(file, tokenizer, max_length=1536, max_cnt=300, shuffle=True), shuffle=True
        )
    validation_dataset = Dataset(validation_files[0], tokenizer, max_length=1280)
    for file in validation_files[1:]:
        validation_dataset.merge(
            Dataset(file, tokenizer, max_length=1280)
        )
    
    training_sampler = torch.utils.data.SequentialSampler(training_dataset)
    validation_sampler = torch.utils.data.SequentialSampler(validation_dataset)
    training_loader = torch.utils.data.DataLoader(
        dataset=training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=training_sampler, collate_fn=custom_collate
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, batch_size=3*batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=validation_sampler, collate_fn=custom_collate
    )

    optimizer = Adafactor(model.parameters(), lr=2.5e-6, scale_parameter=False, relative_step=False)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer, num_warmup_steps=0, num_training_steps=int(epochs * len(training_loader))
    # )
    for epoch in range(epochs):
        model.train()
        training_loss = []
        start_time = time.time()
        oom = 0
        for i, data in enumerate(training_loader):
            data = {
                'input_ids': data['input_ids'].to(device_ids[0]),
                'labels': data['labels'].to(device_ids[0]),
                'attention_mask': data['attention_mask'].to(device_ids[0])
            }
            try:
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                # output = model(input_ids=data['input_ids'], labels=data['labels'], attention_mask=data['attention_mask'], return_dict=True)
                loss = output.loss

                scaler.scale(loss).mean().backward()
                # loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                scaler.step(optimizer)
                # optimizer.step()
                # scheduler.step()
                scaler.update()
                training_loss.append(loss.mean().item())
            except Exception as e:
                if 'out of memory' in str(e):
                    oom += 1
                model.zero_grad()
                optimizer.zero_grad()
                # scheduler.step()
                del data
                try:
                    del output, loss
                except:
                    pass

                torch.cuda.empty_cache()
                
            if i % 100 == 0 and i > 0:
                torch.cuda.empty_cache()
            if i % 1000 == 0:
                print('epoch: {}, step: {}/{}, loss: {}, oom: {}, time: {}s'.format(
                    epoch + 1, i, len(training_loader),
                    round(sum(training_loss) / len(training_loss), 4),
                    # round(scheduler.get_last_lr()[0], 8), 
                    oom,
                    int(time.time() - start_time)
                ))
                start_time = time.time()
                oom = 0
            if i % 1000 == 0 and i > 0:
                validation_step(model, validation_loader, save_dir)
        validation_step(model, validation_loader, save_dir)


if __name__ == '__main__':
    device_map = {'transformer.wte': 0, 'transformer.ln_f': 0, 'lm_head': 0}
    device_map.update({'transformer.h.' + str(i): 0 for i in range(0, 5)})
    device_map.update({'transformer.h.' + str(i): 1 for i in range(5, 14)})
    device_map.update({'transformer.h.' + str(i): 2 for i in range(14, 23)})
    device_map.update({'transformer.h.' + str(i): 3 for i in range(23, 32)})
    device_ids = [0, 1, 2, 3]
    
    vocabulary_file = 'Salesforce/codegen-2B-mono'
    pretrained_file = '/local2/jiang719/language_models/codegen/backup/'
    finetune(
        training_files = [
            # '../../finetuning_rlhf_result/finetuning_spider_train.jsonl',
            # '../../finetuning_rlhf_result/finetuning_sparc_train.jsonl',
            '../../finetuning_rlhf_result/finetuning_cosql_train.jsonl',
            '../../finetuning_rlhf_result/finetuning_cosql_tmp.jsonl'
        ],
        validation_files = [
            # '../../finetuning_rlhf_result/finetuning_spider_dev.jsonl',
            # '../../finetuning_rlhf_result/finetuning_sparc_dev.jsonl',
            '../../finetuning_rlhf_result/finetuning_cosql_dev.jsonl'
        ], epochs=1, batch_size=2,
        save_dir='/local2/jiang719/language_models/codegen/codegen-2B-sql-cosql/'
    )
