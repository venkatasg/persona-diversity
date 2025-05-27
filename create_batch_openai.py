import os 
import csv
import ipdb
import pandas as pd
import json
from openai import OpenAI
from argparse import ArgumentParser
from copy import deepcopy


with open('../openai.txt', 'r') as f:
    api_key = f.read().strip()
    
client = OpenAI(api_key=api_key)
    
task_template = {
        "custom_id": "",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            # This is what you would have in your Chat Completions API call
            "model": "gpt-4.1-mini-2025-04-14",
            "temperature": 1,
            "messages": [
                {
                    "role": "user",
                    "content": ""
                }
            ],
        }
    }
    
def create_batch(questions_file, personas_file):
    # read in tsv file with pandas
    df = pd.read_csv(questions_file, sep='\t')
    questions = df['instruction'].tolist()
    num_words_list = df['num_words_round']
    
    prompts = []
    
    # Load all no-persona prompts
    # for pers_id in [-1]:
    #     for prompt_id, question in enumerate(questions):
    #         task = deepcopy(task_template)
    #         task['body']['messages'][0]['content'] = f"{question}"
    #         task['custom_id'] = f"np-{pers_id}-{prompt_id}"
    #         prompts.append(task)
    #         
    # # Load all no-persona plus cut-off prompts
    # for pers_id in [-1]:
    #     for prompt_id, question in enumerate(questions):
    #         num_words_round = num_words_list[prompt_id]
    #         task = deepcopy(task_template)
    #         task['body']['messages'][0]['content'] = f"""Respond to the following question/instruction in {num_words_round} words or less:\n\n{question}"""
    #         task['custom_id'] = f"npcu-{pers_id}-{prompt_id}"
    #         prompts.append(task)
    # 
    personas = open(personas_file, "r").read().splitlines()
    
    # Load all persona prompts
    for pers_id, persona in enumerate(personas):
        for prompt_id, question in enumerate(questions):
            task = deepcopy(task_template)
            task['body']['messages'][0]['content'] = f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona:\n\n{question}"
            task['custom_id'] = f"fp-{pers_id}-{prompt_id}"
            prompts.append(task)
    
    # Load all persona prompts with cutoff
    for pers_id, persona in enumerate(personas):
        for prompt_id, question in enumerate(questions):
            num_words_round = num_words_list[prompt_id]
            task = deepcopy(task_template)
            task['body']['messages'][0]['content'] = f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona in {num_words_round} words or less:\n\n{question}"
            task['custom_id'] = f"fpcu-{pers_id}-{prompt_id}"
            prompts.append(task)
    
    file_name = "data/openai-batch/cp.jsonl"
    
    with open(file_name, 'w') as file:
        for obj in prompts:
            file.write(json.dumps(obj) + '\n')

def submit_job(batch_file):
    '''
    Submits batch job to OpenAI client
    '''
    batch_input_file = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    print(batch_input_file)
    batch_input_file_id = batch_input_file.id
    
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "No persona prompts with and without cutoff"
        }
    )
        
def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='prep', choices=['prep', 'submit', 'check', 'retreive'])
    args = parser.parse_args()
    
    if args.mode=='prep':
        create_batch("data/dolly_creative_prompts_sample.tsv", "data/sample_coarse_personas.txt")
    elif args.mode=='submit':
        submit_job('data/openai-batch/cp.jsonl')
    elif args.mode=='check':
        output = client.batches.list(limit=100)
        for batch in output.model_dump()['data']:
            print(batch['status'], batch['created_at'], batch['output_file_id'])
    elif args.mode=='retreive':
        result_file_id = ""
        result = client.files.content(result_file_id).content
        with open("data/openai-batch/fp-outputs.jsonl", 'wb') as file:
            file.write(result)
    
if __name__ == "__main__":
    main()