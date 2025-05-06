import csv
import pandas as pd
import os 
import ipdb
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline, logging as loggingt, BitsAndBytesConfig
from huggingface_hub import list_repo_refs 
from argparse import ArgumentParser
from tqdm.auto import tqdm
import torch
import sys
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
            # logging.StreamHandler()  # This will print to console as well
        ]
    )

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
            for pers_id, persona in enumerate(personas):
                for prompt_id, question in enumerate(questions):
                    num_words_round = num_words_list[prompt_id]
                    prompts.append({'prompt_id': prompt_id, 'persona_id': pers_id, 'prompt': f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona in {num_words_round} words or less:\n\n{question}"})
        
        # Persona prompt with no cutoff specified            
        else:
            for pers_id, persona in enumerate(personas):
                for prompt_id, question in enumerate(questions):
                    prompts.append({'prompt_id': prompt_id, 'persona_id': pers_id, 'prompt': f"Assume you are the following persona: {persona}.\n\nNow respond to the following question/instruction appropriately from the perspective of the above persona:\n\n{question}"})
        
    # Now handle no persona cases    
    else:
        if cutoff:
            for pers_id in [-1]:
                for prompt_id, question in enumerate(questions):
                    num_words_round = num_words_list[prompt_id]
                    prompts.append({'prompt_id': prompt_id, 'persona_id': pers_id, 'prompt': f"Respond to the following question/instruction in {num_words_round} words or less:\n\n{question}"})
        else:
            for pers_id in [-1]:
                for prompt_id, question in enumerate(questions):
                    prompts.append({'prompt_id': prompt_id, 'persona_id': pers_id, 'prompt': f"{question}"})
    
    return prompts

def load_pipe(model_name):
    
    config = AutoConfig.from_pretrained(model_name)
    config.use_flash_attention = True  # Enable flash attention if available
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    
    # Enable torch compile for faster inference
    # logging.info("Enabling torch.compile()")
    # model = torch.compile(model)
        
    return pipeline('text-generation', model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16}, device='cuda:1')


def run_model(raw_prompts, model_name, results_dir, num_iterations=1, question_set="dolly", batch_size=16):
    output_file = f"{results_dir}/{model_name.split('/')[-1]}_{question_set}_output.tsv"
    
    # Setup logging
    log_file = f"{results_dir}/inference.log"
    setup_logging(log_file)

    pipeline = load_pipe(model_name)
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
    logging.info("Pipeline created successfully")
    
    total_prompts = len(raw_prompts)
    logging.info(f"Starting inference on {total_prompts} prompts")

    with open(output_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(['prompt_id', 'persona_id', 'response'])
        ### FIX THE FIRST COLUMN - IT SHOULD HAVE THE INSTRUCTION MAPPED CORRECTLY
        processed_prompts = 0
        for i in range(0, total_prompts, batch_size):
            batch = raw_prompts[i:i + batch_size]
            try:
                prompts = [[{"role": "user", "content": message['prompt']}] for message in batch]
                
                responses = pipeline(
                    prompts,
                    max_new_tokens=1024,
                    batch_size=batch_size,
                    do_sample=True,
                    temperature=1
                )
                # ipdb.set_trace()
                for ind, all_responses in enumerate(responses):
                    for resp in all_responses:
                       writer.writerow([batch[ind]['prompt_id'], batch[ind]['persona_id'], resp['generated_text'][-1]['content']])
                
                processed_prompts += batch_size
                logging.info(f"Progress: {processed_prompts}/{total_prompts} prompts processed")
                
            except Exception as e:
                logging.error(f"Error processing sample: {str(e)}")
                sys.exit()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, default='dolly')
    parser.add_argument("--persona", action='store_true')
    parser.add_argument("--cutoff", action='store_true')
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    
    prompts = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_coarse_personas.txt", args.persona, args.cutoff)   

    run_model(raw_prompts=prompts, model_name=args.model, results_dir=args.output, question_set=args.data, batch_size=args.batch)


if __name__ == "__main__":
    main()