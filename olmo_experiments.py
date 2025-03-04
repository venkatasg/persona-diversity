import csv
import json
import pandas as pd
import os
import torch
import logging
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline, logging as loggingt
from huggingface_hub import list_repo_refs
from argparse import ArgumentParser
from tqdm.auto import tqdm

# Reduce verbosity of transformers
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

def print_gpu_utilization():
    """Print current GPU memory usage"""
    logging.info("GPU Utilization:")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

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
    """
    Load model with multi-GPU support
    """
    logging.info(f"Loading model {model_name} {'with checkpoint '+ckpt if ckpt else ''}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logging.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logging.info(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            logging.info(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        logging.warning("No GPUs available! Using CPU only.")
    
    # Create config with optimizations
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, 'use_flash_attention'):
        config.use_flash_attention = True
        logging.info("Enabled flash attention")
    
    # Load model with device_map="auto" for multi-GPU
    if ckpt:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            revision=ckpt,
            device_map="auto",  # Enable multi-GPU
            torch_dtype=torch.float16  # Use half precision
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=ckpt,
            padding_side='left'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="auto",  # Enable multi-GPU
            torch_dtype=torch.float16  # Use half precision
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add default chat template if none exists
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        logging.info("No chat template found, adding a default template")
        # Simple template that works for many models
        default_template = "{% for message in messages %}"
        default_template += "{% if message['role'] == 'user' %}User: {{ message['content'] }}{% endif %}"
        default_template += "{% if message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}"
        default_template += "{% if message['role'] == 'system' %}System: {{ message['content'] }}{% endif %}"
        default_template += "{% if not loop.last %}\n{% endif %}"
        default_template += "{% endfor %}"
        default_template += "{% if add_generation_prompt %}Assistant: {% endif %}"
        
        tokenizer.chat_template = default_template
        logging.info("Added default chat template")
    
    logging.info("Model loaded successfully")
    print_gpu_utilization()
    
    # Create pipeline with multi-GPU model
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device_map="auto"  # Use the same device mapping as the model
    )
    
    return pipe

def run_model(prompts, model_name, results_dir, num_iterations=3, question_set="subj", ckpt=None, batch_size=1, save_every=10):
    """
    Run model inference with proper TSV and JSON continuation support
    """
    # Set up file paths
    if ckpt:
        base_name = f"{results_dir}/{model_name.split('/')[-1]}_{ckpt}_{question_set}"
    else:
        base_name = f"{results_dir}/{model_name.split('/')[-1]}_{question_set}"
    
    tsv_output = f"{base_name}_output.tsv"
    json_output = f"{base_name}_output.json"
    
    # Setup logging
    log_file = f"{question_set}.log"
    setup_logging(log_file)
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model
    pipeline = load_pipe(model_name, ckpt)
    if hasattr(pipeline.tokenizer, 'pad_token_id') and pipeline.tokenizer.pad_token_id is None:
        if hasattr(pipeline.model.config, 'eos_token_id'):
            if isinstance(pipeline.model.config.eos_token_id, list):
                pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
            else:
                pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
        else:
            pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    
    logging.info("Pipeline created successfully")
    
    # Format prompts for model
    try:
        formatted_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
    except Exception as e:
        logging.error(f"Error formatting prompts: {str(e)}")
        # Fallback to simpler format if formatting fails
        logging.info("Using fallback prompt format")
        formatted_prompts = prompts
    
    total_prompts = len(formatted_prompts)
    logging.info(f"Total prompts to process: {total_prompts}")
    
    # Determine continuation state by checking both files
    tsv_exists = os.path.exists(tsv_output)
    json_exists = os.path.exists(json_output)
    
    # Load or initialize results
    all_results = []
    tsv_row_count = 0
    processed_prompts_set = set()
    
    # SCENARIO 1: Both files exist - verify they're in sync
    if json_exists and tsv_exists:
        logging.info("Both JSON and TSV files exist. Checking consistency...")
        
        # Load JSON results
        try:
            with open(json_output, 'r', encoding='utf-8') as json_file:
                all_results = json.load(json_file)
            logging.info(f"Loaded {len(all_results)} results from {json_output}")
            
            # Build set of processed prompts from JSON
            for result in all_results:
                processed_prompts_set.add(result["prompt"])
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON file {json_output}. Will rebuild from TSV.")
            all_results = []
        
        # Count TSV rows (excluding header)
        with open(tsv_output, 'r', encoding='utf-8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # Skip header
            next(tsv_reader, None)
            tsv_row_count = sum(1 for _ in tsv_reader)
        
        logging.info(f"TSV file contains {tsv_row_count} data rows")
        
        # Check if number of results matches between files
        # Accounting for num_iterations responses per prompt
        expected_tsv_rows = len(all_results)
        if tsv_row_count != expected_tsv_rows:
            logging.warning(f"Mismatch between JSON ({expected_tsv_rows} entries) and TSV ({tsv_row_count} rows)")
            
            # Determine which file has more complete data
            if tsv_row_count > expected_tsv_rows:
                logging.info("TSV file has more data. Rebuilding JSON from TSV...")
                # Will rebuild JSON from TSV below
                all_results = []
            else:
                logging.info("JSON file has more data. Regenerating TSV from JSON...")
                # Will regenerate TSV from JSON below
    
    # SCENARIO 2: Only JSON exists - regenerate TSV
    elif json_exists and not tsv_exists:
        logging.info("JSON exists but TSV is missing. Will regenerate TSV from JSON.")
        try:
            with open(json_output, 'r', encoding='utf-8') as json_file:
                all_results = json.load(json_file)
            logging.info(f"Loaded {len(all_results)} results from {json_output}")
            
            # Build set of processed prompts
            for result in all_results:
                processed_prompts_set.add(result["prompt"])
        except json.JSONDecodeError:
            logging.warning(f"Could not parse JSON file {json_output}. Starting fresh.")
            all_results = []
    
    # SCENARIO 3: Only TSV exists - rebuild JSON
    elif tsv_exists and not json_exists:
        logging.info("TSV exists but JSON is missing. Will rebuild JSON from TSV.")
        # JSON will be rebuilt from TSV below
    
    # SCENARIO 4: Neither exists - start fresh
    else:
        logging.info("No existing files found. Starting fresh.")
    
    # Rebuild JSON from TSV if needed
    if tsv_exists and (not json_exists or len(all_results) == 0):
        logging.info("Rebuilding JSON data from TSV file...")
        all_results = []
        
        with open(tsv_output, 'r', encoding='utf-8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            # Skip header row
            header = next(tsv_reader, None)
            if not header or len(header) < 2:
                logging.warning("TSV file appears to be empty or corrupted. Starting fresh.")
            else:
                for row in tsv_reader:
                    if len(row) >= 2:
                        prompt_content = row[0]
                        response_text = row[1]
                        
                        # Find original prompt in prompts list
                        original_prompt = None
                        for p in prompts:
                            if isinstance(formatted_prompts[0][0], dict):
                                # Handle nested dict format
                                if prompt_content == p:
                                    original_prompt = p
                                    break
                            elif prompt_content in p:
                                original_prompt = p
                                break
                        
                        # If we couldn't find it, use the content as is
                        if not original_prompt:
                            original_prompt = prompt_content
                        
                        all_results.append({"prompt": original_prompt, "response": response_text})
                        processed_prompts_set.add(original_prompt)
                
                logging.info(f"Rebuilt {len(all_results)} results from TSV")
    
    # Regenerate TSV from JSON if needed
    if json_exists and len(all_results) > 0 and (not tsv_exists or tsv_row_count < len(all_results)):
        logging.info("Regenerating TSV from JSON data...")
        with open(tsv_output, 'w', newline='', encoding='utf-8') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t')
            tsv_writer.writerow(['prompt', 'response'])  # Header
            
            for result in all_results:
                prompt_content = result["prompt"]
                response_text = result["response"]
                
                # For formatted prompts, get the content
                if isinstance(formatted_prompts[0][0], dict):
                    for fp in formatted_prompts:
                        if fp[0]['content'] == prompt_content:
                            prompt_content = fp[0]['content']
                            break
                
                tsv_writer.writerow([prompt_content, response_text])
            
            logging.info(f"Regenerated TSV with {len(all_results)} rows")
    
    # Calculate how many prompts we've already processed
    processed_count = len(processed_prompts_set)
    logging.info(f"Already processed {processed_count} unique prompts")
    
    # Skip prompts we've already processed
    prompts_to_process = []
    formatted_prompts_to_process = []
    
    for i, prompt in enumerate(prompts):
        if prompt not in processed_prompts_set:
            prompts_to_process.append(prompt)
            formatted_prompts_to_process.append(formatted_prompts[i])
    
    if len(prompts_to_process) < len(prompts):
        logging.info(f"Skipping {len(prompts) - len(prompts_to_process)} already processed prompts")
        prompts = prompts_to_process
        formatted_prompts = formatted_prompts_to_process
    
    # If all prompts are processed, we're done
    if not prompts:
        logging.info("All prompts have already been processed. Nothing to do.")
        return
    
    logging.info(f"Continuing with {len(prompts)} remaining prompts")
    
    # Open TSV file in append mode if it exists, otherwise create new
    tsv_mode = 'a' if tsv_exists else 'w'
    with open(tsv_output, tsv_mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="\t")
        # Write header only if creating a new file
        if tsv_mode == 'w':
            writer.writerow(['prompt', 'response'])
        
        total_responses = 0
        save_counter = 0
        
        # Process in batches
        effective_batch_size = max(1, min(batch_size, 32))  # Cap batch size to avoid OOM
        
        # Main processing loop
        for i in range(0, len(formatted_prompts), effective_batch_size):
            batch_prompts = formatted_prompts[i:i + effective_batch_size]
            original_prompts = prompts[i:i + effective_batch_size]
            
            try:
                # Run inference
                try:
                    # First try with chat-formatted prompts
                    responses = pipeline(
                        batch_prompts,
                        max_new_tokens=256,
                        num_return_sequences=num_iterations,
                        do_sample=True,
                        temperature=0.7
                    )
                except ValueError as e:
                    if "chat template" in str(e).lower():
                        # If chat template error occurs, try with plain text prompts
                        logging.warning("Chat template error. Trying with plain text prompts.")
                        plain_prompts = [p[0]['content'] if isinstance(p[0], dict) else p for p in batch_prompts]
                        responses = pipeline(
                            plain_prompts,
                            max_new_tokens=256,
                            num_return_sequences=num_iterations,
                            do_sample=True,
                            temperature=0.7
                        )
                    else:
                        raise e
                
                # Process all responses
                batch_results = []
                
                # If only one prompt in batch, wrap responses in a list
                if len(batch_prompts) == 1:
                    responses = [responses]
                
                for j, responses_for_prompt in enumerate(responses):
                    original_prompt = original_prompts[j]
                    
                    # Get prompt content for TSV
                    if isinstance(batch_prompts[j][0], dict) and 'content' in batch_prompts[j][0]:
                        prompt_content = batch_prompts[j][0]['content']
                    else:
                        prompt_content = str(batch_prompts[j])
                    
                    # Handle each of the iterations for this prompt
                    for resp in responses_for_prompt:
                        try:
                            # Extract the response content
                            try:
                                if isinstance(resp['generated_text'], list):
                                    response_text = resp['generated_text'][-1]['content']
                                elif isinstance(resp, dict) and 'generated_text' in resp:
                                    response_text = resp['generated_text']
                                elif isinstance(resp, str):
                                    response_text = resp
                                else:
                                    if hasattr(resp, 'text'):
                                        response_text = resp.text
                                    elif hasattr(resp, 'content'):
                                        response_text = resp.content
                                    else:
                                        response_text = str(resp)
                            except Exception as extract_error:
                                logging.error(f"Error extracting response: {str(extract_error)}")
                                logging.error(f"Response type: {type(resp)}")
                                logging.error(f"Response preview: {str(resp)[:100]}...")
                                response_text = str(resp)
                            
                            writer.writerow([prompt_content, response_text])
                            batch_results.append({"prompt": original_prompt, "response": response_text})
                            total_responses += 1
                            save_counter += 1
                        except (KeyError, IndexError) as e:
                            logging.error(f"Error extracting response: {str(e)} for response: {resp}")
                            continue
                
                # Add results to master list
                all_results.extend(batch_results)
                
                # Save periodically
                if save_counter >= save_every:
                    with open(json_output, 'w', encoding='utf-8') as json_file:
                        json.dump(all_results, json_file, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {len(all_results)} results to {json_output}")
                    save_counter = 0
                    file.flush()
                
                logging.info(f"Progress: {i + len(batch_prompts) + processed_count}/{total_prompts} prompts processed, {total_responses} new responses")
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logging.error(f"OOM error with batch size {effective_batch_size}. Try reducing batch size.")
                    # Save progress
                    file.flush()
                    with open(json_output, 'w', encoding='utf-8') as json_file:
                        json.dump(all_results, json_file, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {len(all_results)} results before error.")
                    
                    # Try with smaller batch size if possible
                    if effective_batch_size > 1:
                        effective_batch_size = max(1, effective_batch_size // 2)
                        logging.info(f"Reducing batch size to {effective_batch_size} and continuing...")
                        continue
                    else:
                        raise e
                else:
                    # Other errors
                    logging.error(f"Error during generation: {str(e)}")
                    # Save progress
                    file.flush()
                    with open(json_output, 'w', encoding='utf-8') as json_file:
                        json.dump(all_results, json_file, ensure_ascii=False, indent=2)
                    logging.info(f"Saved {len(all_results)} results before error.")
                    continue
        
        # Final save
        with open(json_output, 'w', encoding='utf-8') as json_file:
            json.dump(all_results, json_file, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(all_results)} results to {json_output} (final save)")

def process_all_datasets(model_name, ckpt=None, batch_size=1, save_every=10):
    """
    Process both datasets (subjective and dolly) sequentially
    """
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    print(f"Processing model: {model_name} {'with checkpoint '+ckpt if ckpt else ''}")
    
    # Process subjective dataset
    print("\n\n======= PROCESSING SUBJECTIVE DATASET =======")
    prompts = create_prompts_subjective("data/subj_questions.txt", "data/sample_personas.txt")
    run_model(
        prompts=prompts,
        model_name=model_name,
        results_dir="output",
        num_iterations=3,
        question_set="subj",
        ckpt=ckpt,
        batch_size=batch_size,
        save_every=save_every
    )
    
    # Clear CUDA cache between datasets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nCleared CUDA cache")
    
    # Process dolly dataset
    print("\n\n======= PROCESSING DOLLY DATASET =======")
    prompts_dolly = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_personas.txt")
    run_model(
        prompts=prompts_dolly,
        model_name=model_name,
        results_dir="output",
        num_iterations=3,
        question_set="dolly",
        ckpt=ckpt,
        batch_size=batch_size,
        save_every=save_every
    )
    
    print(f"\nCompleted processing all datasets for {model_name}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data", type=str, help="Dataset to use: 'dolly', 'subj', or 'all'")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--save_every", type=int, default=10, help="Save results every N responses")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint to use")
    parser.add_argument("--all_datasets", action="store_true", help="Process all datasets sequentially")
    args = parser.parse_args()
    
    # Check if all_datasets is specified or data is set to 'all'
    if args.all_datasets or args.data == 'all':
        process_all_datasets(args.model, args.ckpt, args.batch_size, args.save_every)
    else:
        # Check GPUs
        if torch.cuda.is_available():
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        else:
            print("No GPUs available!")
        
        # If not processing all datasets, process the specified one
        data_type = args.data if args.data else "subj"  # Default to subjective if not specified
        
        # Load prompts based on dataset
        if data_type == 'dolly':
            prompts = create_prompts_dolly("data/dolly_creative_prompts_sample.tsv", "data/sample_personas.txt")
        else:
            prompts = create_prompts_subjective("data/subj_questions.txt", "data/sample_personas.txt")
        
        # Run model
        run_model(
            prompts=prompts,
            model_name=args.model,
            results_dir="output",
            num_iterations=3,
            question_set=data_type,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            save_every=args.save_every
        )

if __name__ == "__main__":
    main()