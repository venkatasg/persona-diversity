import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import ipdb
import re
from datasets import load_dataset
from argparse import ArgumentParser
from diversity import homogenization_score
import markdown
import emoji
from bs4 import BeautifulSoup
from transformers.utils import logging
logging.set_verbosity_error() 

# Set random seeds for reproducibility on a specific machine
random.seed(1)
np.random.seed(1)
np.random.RandomState(1)
np.set_printoptions(precision=3)
    
def unformat(text):
    html = markdown.markdown(text.strip())
    soup = BeautifulSoup(html, features='html.parser')
    unformatted_text = soup.get_text()
    final_text = emoji.replace_emoji(unformatted_text, replace='').replace('\n', ' ')
    return final_text
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--datafile", type=str, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_shuffles", type=int, default=3)
    parser.add_argument("--noverb", action='store_false')
    parser.add_argument("--unformat", action='store_true')
    args = parser.parse_args()
    
    if args.datafile=='dolly':
        dolly = load_dataset("databricks/databricks-dolly-15k")["train"].filter(lambda row: row['category']=='creative_writing').to_pandas()
        sample = pd.read_csv('data/dolly_creative_prompts_sample.tsv', sep='\t')
        sample['response'] = sample['index'].apply(lambda x: dolly.loc[x, 'response'])
        sample['prompt_id'] = [i for i in range(len(sample))]
        
        print(np.round(homogenization_score(sample.response.values.tolist(), measure='bertscore', batch_size=args.batch, verbose=args.noverb, model="microsoft/deberta-base-mnli"), 2)) 
    else:
        df = pd.read_csv(args.datafile, sep='\t')
        
        if 'persona_id' not in df.columns:
            if df.shape[0]>100:
                prompt_ids = []
                persona_ids = []
                for pr_id in range(100):
                    prompt_ids += [pr_id for _ in range(100)]
                    persona_ids += [j for j in range(100)]
                df['persona_id'] = persona_ids
                df['prompt_id'] = prompt_ids
            else:
                df['persona_id'] = [-1 for i in range(100)]
                df['prompt_id'] = [i for i in range(100)]
        
        if args.unformat:
            df['response'] = df.response.apply(lambda x: unformat(x))
        else:
            df['response'] = df.response.apply(lambda x: x.strip())
        
        if df.persona_id.unique().shape[0]==1:
            data = df.response.values.tolist()
            print(np.round(homogenization_score(data, measure='bertscore', batch_size=args.batch, verbose=args.noverb, model="microsoft/deberta-base-mnli"), 2))
        else:
            bs_scores = []
            random.seed(1)
            for _ in range(args.num_shuffles):
                persona_ids_shuffled = [i for i in range(100)]
                random.shuffle(persona_ids_shuffled)
                prompt_ids = [i for i in range(100)]
                pairs = list(zip(persona_ids_shuffled, prompt_ids))
                
                new_df = df.set_index(['persona_id', 'prompt_id'])
                data = new_df.loc[pairs, 'response'].values.tolist()
                bs_scores.append(homogenization_score(data, measure='bertscore', batch_size=args.batch, verbose=args.noverb, model="microsoft/deberta-base-mnli"))
            print(bs_scores)
            print(f"Mean:{np.round(np.mean(bs_scores),2)}, SD: {np.round(np.std(bs_scores),2)}")
    
if __name__ == "__main__":
    main()