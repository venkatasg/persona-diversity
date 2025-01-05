import csv
import pandas as pd
import os 
import ipdb
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline, logging as loggingt
from huggingface_hub import list_repo_refs 
from argparse import ArgumentParser
from tqdm.auto import tqdm
import torch

loggingt.set_verbosity_error() 

def setup_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will print to console as well
        ]
    )

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
    
    config = AutoConfig.from_pretrained(model_name)
    config.use_flash_attention = True  # Enable flash attention if available
    
    if ckpt: 
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, revision=ckpt)
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=ckpt, padding_side='left')
    else: 
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    # Enable torch compile for faster inference
    logging.info("Enabling torch.compile()")
    model = torch.compile(model)
        
    return pipeline('text-generation', model=model, tokenizer=tokenizer, device="cuda", torch_dtype='float16')


def run_model(prompts, model_name, results_dir, num_iterations=3, question_set="subj", ckpt=None, batch_size=8):
    if ckpt: 
        output_file = f"{results_dir}/{model_name.split('/')[-1]}_{ckpt}_{question_set}_output.tsv"
    else: 
        output_file = f"{results_dir}/{model_name.split('/')[-1]}_{question_set}_output.tsv"
    
    # Setup logging
    log_file = f"{question_set}.log"
    setup_logging(log_file)

    pipeline = load_pipe(model_name, ckpt)
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
    logging.info("Pipeline created successfully")
    
    prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    total_prompts = len(prompts)
    logging.info(f"Starting inference on {total_prompts} prompts")

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(['prompt', 'response'])
        
        processed_prompts = 0
        for i in range(0, total_prompts, batch_size):
            batch = prompts[i:i + batch_size]
            try:
                responses = pipeline(
                    batch,
                    max_new_tokens=256,
                    batch_size=batch_size,
                    num_return_sequences=num_iterations,
                    do_sample=True,
                    temperature=0.7
                )
                
                for i, responses_3 in enumerate(responses):
                    for resp in responses_3:
                        writer.writerow([batch[i][0]['content'], resp['generated_text'][-1]['content']])
                
                processed_prompts += len(batch)
                logging.info(f"Progress: {processed_prompts}/{total_prompts} prompts processed")
                
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                continue


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    
    if args.data=='dolly':
        prompts = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_personas.txt")
        
    else:
        prompts = create_prompts_subjective("data/subj_questions.txt", "data/sample_personas.txt")
    
    run_model(prompts, args.model, "output", num_iterations=3, question_set=args.data)


if __name__ == "__main__":
    main()