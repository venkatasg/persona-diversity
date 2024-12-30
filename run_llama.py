import csv
import pandas as pd
import os 
import ipdb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import list_repo_refs 
from argparse import ArgumentParser

from tqdm import tqdm

def create_prompts_subjective(questions_file, personas_file):
    questions = open(questions_file, "r").read().splitlines()
    personas = open(personas_file, "r").read().splitlines()
    prompts = []
    for persona in personas:
        for question in questions:
            prompts.append("Imagine you are {}. {}".format(persona, question))
    return prompts

def create_prompts_dolly(questions_file, personas_file):
    # read in tsv file with pandas
    questions = pd.read_csv(questions_file, sep='\t')
    questions = questions['instruction'].tolist()
    personas = open(personas_file, "r").read().splitlines()
    prompts = []
    for persona in personas:
        for question in questions:
            prompts.append(f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona:\n\n{question}")
    return prompts

def load_pipe(model_name, ckpt=None):
    if ckpt: 
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=ckpt)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=ckpt, padding_side='left')
    else: 
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    return pipeline('text-generation', model=model, tokenizer=tokenizer, device_map="auto")


def run_model(prompts, model_name, results_dir, num_iterations=3, question_set="subj", ckpt=None):
    if ckpt: 
        output_file = f"{results_dir}/{model_name.split('/')[-1]}_{ckpt}_{question_set}_output.csv"
    else: 
        output_file = f"{results_dir}/{model_name.split('/')[-1]}_{question_set}_output.csv"

    pipeline = load_pipe(model_name, ckpt, torch_dtype='float16')

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['prompt', 'response'])
        
        
        if 'instruct' in model_name.lower(): 
            prompts = [{"role": "user", "content": prompt} for prompt in prompts]
                
        for n in range(num_iterations):
            response = pipeline(prompts, batch_size=2, max_new_tokens=256, num_return_sequences=1, do_sample=True, temperature=0.7)
            if 'instruct' in model_name.lower():
                for i, resp in enumerate(response):
                    writer.writerow([prompts[i]['content'], resp])
            else:
                for i, resp in enumerate(response):
                    writer.writerow([prompts[i], resp[0]['generated_text']]) 
            
            print(f"Prompt {i+1}/{len(prompts)} completed")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    
    prompts = create_prompts_subjective("data/subj_questions.txt", "data/sample_personas.txt")
    prompts_dolly = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_personas.txt")
    ipdb.set_trace()
    run_model(prompts, args.model, "output", num_iterations=3, question_set="subj")
    run_model(prompts_dolly, args.model, "output", num_iterations=3, question_set="dolly")


if __name__ == "__main__":
    main()