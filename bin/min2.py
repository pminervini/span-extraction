#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers.data.processors.squad import SquadResult, SquadV2Processor

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import squad_convert_examples_to_features
from transformers import get_linear_schedule_with_warmup, AdamW

from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate


def to_list(tensor):
    return tensor.detach().cpu().tolist()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForQuestionAnswering.from_pretrained('roberta-base', config=config).to(device)

processor = SquadV2Processor()

train_examples = processor.get_train_examples(None, filename='data/squad-v2.0/train-v2.0.json')
dev_examples = processor.get_dev_examples(None, filename='data/squad-v2.0/dev-v2.0.json')

cached_features_file = 'squad2_train_cache.pt'

if os.path.exists(cached_features_file):
    features_and_dataset = torch.load(cached_features_file)
    features, dataset, examples = (
        features_and_dataset["features"],
        features_and_dataset["dataset"],
        features_and_dataset["examples"])
else:
    features, dataset = squad_convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=True,
        return_dataset="pt",
        threads=4)
    torch.save({"features": features, "dataset": dataset, "examples": train_examples}, cached_features_file)

train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=4)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [{
    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    "weight_decay": 0.0}, {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

t_total = len(train_dataloader) * 1

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

model.zero_grad()

loss_values = []

for epoch_id in range(1):
    for i, batch in enumerate(train_dataloader):
        model.train()

        batch = tuple(t.to(device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        del inputs["token_type_ids"]

        outputs = model(**inputs)

        loss = outputs[0]

        loss.backward()

        loss_values += [loss.item()]

        if (i + 1) % 100 == 0:
            mean = np.mean(loss_values)
            std = np.std(loss_values)
            print(f'{epoch_id}:{i}\t{mean} ± {std}')
            loss_values = []

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        model.zero_grad()

        if i > 1000:
            break

dev_examples = processor.get_dev_examples(None, filename='data/squad-v2.0/dev-v2.0.json')

dev_features, dev_dataset = squad_convert_examples_to_features(
    examples=dev_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    return_dataset="pt",
    threads=32)

eval_sampler = SequentialSampler(dev_dataset)
eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=16)

all_results = []

for batch in tqdm(eval_dataloader):
    model.eval()
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2]
        }
        del inputs["token_type_ids"]

        feature_indices = batch[3]

        outputs = model(**inputs)

    for i, feature_index in enumerate(feature_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)

        output = [to_list(output[i]) for output in outputs]

        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)

        all_results += [result]

predictions = compute_predictions_logits(
    dev_examples,
    features,
    all_results,
    20,
    30,
    True,
    'predictions.json',
    'nbest.json',
    'null_odds.json',
    True,
    True,
    0.0,
    tokenizer)

results = squad_evaluate(dev_examples, predictions)

print(results)
