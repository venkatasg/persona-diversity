import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import csv
import pandas as pd
import ipdb
from together import Together
from argparse import ArgumentParser
import logging
from transformers import logging as loggingt
from tqdm.auto import tqdm
from time import sleep



with open('../together.txt', 'r') as f:
    api_key = f.read().strip()
    
client = Together(
    api_key = api_key
)

# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

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


def run_model(raw_prompts, model_name, results_dir, num_iterations=1, question_set="dolly", ckpt=None, batch_size=16):
    if ckpt: 
        output_file = f"{results_dir}/{model_name.split('/')[-1]}_{ckpt}_{question_set}_output.tsv"
    else: 
        output_file = f"{results_dir}/{model_name.split('/')[-1]}_{question_set}_output.tsv"
    
    # Setup logging
    log_file = f"{question_set}.log"
    setup_logging(log_file)
    
    prompts = [[{"role": "user", "content": prompt['prompt']}] for prompt in raw_prompts]
    total_prompts = len(prompts)
    logging.info(f"Starting inference on {total_prompts} prompts")

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(['prompt_id', 'persona_id', 'response'])
        
        processed_prompts = 0
        for i in range(0, total_prompts):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=prompts[i],
                    temperature=0.7,
                    seed=1,
                    max_tokens=1024
                )
                writer.writerow([raw_prompts[i]['prompt_id'], raw_prompts[i]['persona_id'], response.choices[0].message.content])
                
                processed_prompts += 1
                logging.info(f"Progress: {processed_prompts}/{total_prompts} prompts processed")
                
            except Exception as e:
                logging.error(f"Error processing sample: {str(e)}")
                continue
            if i%60==0:
                sleep(10)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
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