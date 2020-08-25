#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import jsonlines


squad_like = {
    "version": "v2.0",
    "data": []
}

with jsonlines.open('data/nq/filtered/train.jsonl') as f:
    for entry in f.iter():
        print(entry)
        example_id = entry['example_id']
        question = entry['question']
        context = entry['context']
        answers = entry['answers']

        assert len(answers) > 0

        for i, answer in enumerate(answers):

            squad_like_entry = {
                'title': f'Example {example_id}:{i}',
                'paragraphs': [
                    {
                        'qas': [
                            {
                                # 'question': question,
                                'question': '',
                                'id': f'{example_id}:{i}',
                                'answers': [
                                    {
                                        'text': answer,
                                        'answer_start': context.index(answer)
                                    }
                                ],
                                "is_impossible": False
                            }
                        ],
                        'context': context
                    }
                ]
            }

            squad_like['data'] += [squad_like_entry]

with open("/dev/stdout", "w") as outfile:
    json.dump(squad_like, outfile)
