import csv
import pandas as pd
import os 

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from run_bedrock_experiments import create_prompts_subjective, create_prompts_dolly
from huggingface_hub import list_repo_refs 
from argparse import ArgumentParser

def generate_olmo_response(prompt, model_name, ckpt):
    prompt = [{"role": "user", 
               "content": prompt}]
    
    if ckpt: 
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=ckpt, cache_dir="/scratch/shaib.c/")
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=ckpt)
    else: 
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/scratch/shaib.c/")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    response = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)
    
    return response[0]['generated_text']


def run_model(prompts, model, results_dir, num_iterations=3, question_set="subj", ckpt=None):
    if ckpt: 
        output_file = f"{results_dir}/{model.split('/')[-1]}_{ckpt}_{question_set}_output.csv"
    else: 
        output_file = f"{results_dir}/{model.split('/')[-1]}_{question_set}_output.csv"
        
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        
        writer = csv.writer(file)
        writer.writerow(['prompt', 'response'])
        
        for i, prompt in enumerate(prompts):
            for n in range(num_iterations):
                response = generate_olmo_response(prompt, model, ckpt)
                writer.writerow([prompt, response])
                
            print(f"Prompt {i+1}/{len(prompts)} completed")

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B")
    args = parser.parse_args()
    
    os.environ["HF_HOME"] = "/scratch/shaib.c/"
    
    prompts = create_prompts_subjective("data/subj_questions.txt", "data/sample_personas.txt")
    prompts_dolly = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_personas.txt")
    
    if not args.model.startswith('allenai/OLMo'):
        run_model(prompts, "allenai/OLMo-2-1124-7B", "output", num_iterations=3, question_set="subj", ckpt=args.model)
        run_model(prompts_dolly, "allenai/OLMo-2-1124-7B", "output", num_iterations=3, question_set="dolly", ckpt=args.model)
    else:    
        run_model(prompts, args.model, "output", num_iterations=3, question_set="subj")
        run_model(prompts_dolly, args.model, "output", num_iterations=3, question_set="dolly")


if __name__ == "__main__":
    main()