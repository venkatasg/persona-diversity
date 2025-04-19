import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import ipdb
import re
from evaluate import load
from datasets import load_dataset
from argparse import ArgumentParser

# Set random seeds for reproducibility on a specific machine
random.seed(1)
np.random.seed(1)
np.random.RandomState(1)
np.set_printoptions(precision=3)

scorer = load("bertscore")

def _calculate_score(ps, rs, batch_size):
    """ 
    Returns the score of two strings 
    """
    score = scorer.compute(predictions=ps, 
                           references=rs, 
                           model_type="microsoft/deberta-base-mnli",
                           batch_size=batch_size)['f1']
    # return sum of all f1 scores which is averaged later
    return score
    
def hom_bs(data, batch_size=64, verbose=True):

    corpus_score = 0
    doc_score = 0
    
    print('==> Scoring all pairs')
     
    for i, pred  in tqdm(enumerate(data), total=len(data), disable=(not verbose)):
        refs = [x for j,x in enumerate(data) if j!=i]
        preds = [pred for _ in range(len(refs))]
        
        doc_score = sum(_calculate_score(preds, refs, batch_size=batch_size))
        
        corpus_score += doc_score / (len(data) - 1)
        doc_score = 0
    
    # case where all strings are the exact same in the list
    if corpus_score == 0: 
        corpus_score += len(data)
    
    # returns corpus level homogenization score 
    return round(corpus_score/len(data), 3)
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--datafile", type=str, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--num_shuffles", type=int, default=3)
    parser.add_argument("--noverb", action='store_false')
    args = parser.parse_args()
    
    if args.datafile=='dolly':
        dolly = load_dataset("databricks/databricks-dolly-15k")["train"].filter(lambda row: row['category']=='creative_writing').to_pandas()
        sample = pd.read_csv('data/dolly_creative_prompts_sample.tsv', sep='\t')
        sample['response'] = sample['index'].apply(lambda x: dolly.loc[x, 'response'])
        sample['prompt_id'] = [i for i in range(len(sample))]
        
        print(hom_bs(sample.response.values.tolist(), batch_size=args.batch, verbose=args.noverb)) 
    else:
        df = pd.read_csv(args.datafile, sep='\t')
        df['response'] = df.response.apply(lambda x: x.strip())
        
        if df.persona_id.unique().shape[0]==1:
            data = df.response.values.tolist()
            print(hom_bs(data, batch_size=args.batch, verbose=args.noverb))
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
                bs_scores.append(hom_bs(data, batch_size=args.batch, verbose=args.noverb))
            print(bs_scores)
            print(f"Mean:{np.round(np.mean(bs_scores),2)}, SD: {np.round(np.std(bs_scores),2)}")
    
if __name__ == "__main__":
    main()