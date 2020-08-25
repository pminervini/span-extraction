#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

with open('train-v2.0.json') as f:
    data = json.load(f)

print(data['data'][0]['title'])

print(data['data'][0]['paragraphs'][0]['context'])

print(data['data'][0]['paragraphs'][0]['qas'][0])
