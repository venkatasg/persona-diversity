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
import bitsandbytes

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

def create_prompts_dolly(questions_file, personas_file, include_personas: bool, cutoff:bool):
    # read in tsv file with pandas
    df = pd.read_csv(questions_file, sep='\t')
    questions = df['instruction'].tolist()
    num_words_list = df['num_words_round']
    
    prompts = []
    if include_personas:
        
        personas = open(personas_file, "r").read().splitlines()
        
        # Persona prompts with cutoff specified in prompt
        if cutoff:
            for persona in personas:
                for i, question in enumerate(questions):
                    num_words_round = num_words_list[i]
                    prompts.append(f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona in {num_words_round} words or less:\n\n{question}")
        
        # Persona prompt with no cutoff specified            
        else:
            for persona in personas:
                for i, question in enumerate(questions):
                    prompts.append(f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona:\n\n{question}")
        
    # Now handle no persona cases    
    else:
        if cutoff:
            for i, question in enumerate(questions):
                num_words_round = num_words_list[i]
                prompts.append(f"Respond to the following question/instruction in {num_words_round} words or less:\n\n{question}")
        else:
            for i, question in enumerate(questions):
                prompts.append(f"{question}")
    
    return prompts

def load_pipe(model_name):
    
    config = AutoConfig.from_pretrained(model_name)
    config.use_flash_attention = True  # Enable flash attention if available
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if '70B' in model_name:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config,, torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    
    
        
    # Enable torch compile for faster inference
    logging.info("Enabling torch.compile()")
    model = torch.compile(model)
        
    return pipeline('text-generation', model=model, tokenizer=tokenizer, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})


def run_model(raw_prompts, model_name, results_dir, num_iterations=1, question_set="dolly", batch_size=16):
    output_file = f"{results_dir}/{model_name.split('/')[-1]}_{question_set}_output.tsv"
    
    # Setup logging
    log_file = f"{question_set}.log"
    setup_logging(log_file)

    pipeline = load_pipe(model_name)
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
    logging.info("Pipeline created successfully")
    
    prompts = [[{"role": "user", "content": prompt}] for prompt in raw_prompts]
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
                    max_new_tokens=1024,
                    batch_size=batch_size,
                    num_return_sequences=num_iterations,
                    do_sample=True,
                    temperature=0.7
                )
                
                for i, all_responses in enumerate(responses):
                    for resp in all_responses:
                        writer.writerow([prompts[i][0]['content'], resp['generated_text'][-1]['content']])
                
                processed_prompts += batch_size
                logging.info(f"Progress: {processed_prompts}/{total_prompts} prompts processed")
                
            except Exception as e:
                logging.error(f"Error processing sample: {str(e)}")
                continue


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, default='dolly')
    parser.add_argument("--persona", action='store_true')
    parser.add_argument("--cutoff", action='store_true')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    
    if args.data=='dolly':
        prompts = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_personas.txt", args.persona, args.cutoff)   
    elif args.data=='subj':
        prompts = create_prompts_subjective("data/subj_questions.txt", "data/sample_personas.txt")
    # ipdb.set_trace()
    run_model(raw_prompts=prompts, model_name=args.model, results_dir=args.output, question_set=args.data, batch_size=args.batch)


if __name__ == "__main__":
    main()