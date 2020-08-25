#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import json
import jsonlines

import argparse

from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def to_squad_entry(example_id: str,
                   question: str,
                   context: str,
                   answer: str) -> Dict:
    answer_start = None
    if answer in context:
        answer_start = context.index(answer)

    res = {
        'title': f'Example {example_id}',
        'paragraphs': [
            {
                'qas': [
                    {
                        # 'question': question,
                        'question': '',
                        'id': f'{example_id}',
                        'answers': [],
                        "is_impossible": answer_start is None
                    }
                ],
                'context': context
            }
        ]
    }

    if answer_start is not None:
        res['paragraphs'][0]['qas'][0]['answers'] += [{'text': answer, 'answer_start': answer_start}]

    return res


def main(argv):
    parser = argparse.ArgumentParser('NQ2SQUaD2', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sentence', '-s', action='store_true', default=False)

    parser.add_argument('--input', '-i', action='store', type=str, default='data/nq/filtered/train.jsonl')
    parser.add_argument('--output', '-o', action='store', type=str, default='/dev/stdout')

    args = parser.parse_args(argv)

    is_sentence = args.sentence
    input_path = args.input
    output_path = args.output

    if is_sentence:
        nltk.download('punkt')

    squad_like = {
        "version": "v2.0",
        "data": []
    }

    with jsonlines.open(input_path) as f:
        for entry in f.iter():
            example_id = entry['example_id']
            question = entry['question']
            context = entry['context']
            answers = entry['answers']

            assert len(answers) > 0

            for i, answer in enumerate(answers):
                if is_sentence is True:
                    for j, sentence in enumerate(sent_tokenize(context)):
                        squad_like_entry = to_squad_entry(f'{example_id}:{i}:{j}', question, sentence, answer)
                        squad_like['data'] += [squad_like_entry]
                else:
                    squad_like_entry = to_squad_entry(f'{example_id}:{i}', question, context, answer)
                    squad_like['data'] += [squad_like_entry]

    with open(output_path, "w") as outfile:
        json.dump(squad_like, outfile)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main(sys.argv[1:])
