#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

import nlp
import optuna

from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertTokenizerFast
from transformers import BertForQuestionAnswering, AdamW

metric = nlp.load_metric('squad_v2')
dataset = nlp.load_dataset('squad_v2')

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

print(dataset['train'].column_names)

for n, d in dataset.items():
    d.drop('id')
    d.flatten()

print(dataset['train'].column_names)

dataset = dataset.map(lambda example: tokenizer.batch_encode_plus(example['context']), batched=True)

print(dataset['train'].column_names)


def convert_to_features(batch):
    # Tokenize contexts and questions (as pairs of inputs)
    # keep offset mappings for evaluation
    input_pairs = list(zip(batch['context'], batch['question']))
    encodings = tokenizer.batch_encode_plus(input_pairs, padding='longest', return_offsets_mapping=True)

    # Compute start and end tokens for labels
    start_positions, end_positions = [], []

    for i, (text, start) in enumerate(zip(batch['answers.text'], batch['answers.answer_start'])):
        start_pos = end_pos = -1
        if len(start) > 0:
            first_char = start[0]
            last_char = first_char + len(text[0]) - 1
            start_pos = encodings.char_to_token(i, first_char)
            end_pos = encodings.char_to_token(i, last_char)
        start_positions.append(start_pos)
        end_positions.append(end_pos)

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
    return encodings


dataset = dataset.map(convert_to_features, batched=True)

print(dataset['train'].column_names)

# Format our dataset to outputs torch.Tensor to train a pytorch model
columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
dataset['train'].set_format(type='torch', columns=columns)

# Instantiate a PyTorch Dataloader around our dataset
dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=8)

print(dataset['train'].column_names)

model = BertForQuestionAnswering.from_pretrained('bert-base-cased')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()
for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    if i > 3:
        break
