#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler

from transformers.data.processors.squad import SquadResult, SquadV2Processor

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import squad_convert_examples_to_features
from transformers import get_linear_schedule_with_warmup, AdamW

config = AutoConfig.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModelForQuestionAnswering.from_pretrained('roberta-base', config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

processor = SquadV2Processor()

train_examples = processor.get_train_examples(None, filename='data/squad-v2.0/train-v2.0.json')
dev_examples = processor.get_dev_examples(None, filename='data/squad-v2.0/dev-v2.0.json')

features, dataset = squad_convert_examples_to_features(
    examples=train_examples,
    tokenizer=tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=True,
    return_dataset="pt",
    threads=4)

cached_features_file = 'squad2_train_cache.pt'

torch.save({"features": features, "dataset": dataset, "examples": train_examples}, cached_features_file)

train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=4)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [{
    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    "weight_decay": 0.0}, {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

t_total = len(train_dataloader) * 3

optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

model.zero_grad()

