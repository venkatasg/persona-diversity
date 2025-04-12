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
        # refs = [refs[b:b+batch_size] for b in range(0, len(data)-1, batch_size)]
        # preds = [[pred for _ in range(len(refs[b_ind]))] for b_ind in range(len(refs))]
        preds = [pred for _ in range(len(refs))]
        
        # for batch_ind in range(len(refs)):
        #     doc_score += sum(_calculate_score(preds[batch_ind], refs[batch_ind], batch_size=batch_size))
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
    parser.add_argument("--data", type=str, default='dolly')
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--noverb", action='store_false')
    args = parser.parse_args()
    
    if args.data=='dolly':
        data = load_dataset("databricks/databricks-dolly-15k")["train"].to_pandas().instruction.sample(args.samples, random_state=1).values.tolist()
    elif args.data=='personahub':
        data = load_dataset("proj-persona/PersonaHub", "instruction")['train'].to_pandas()['synthesized text'].sample(args.samples, random_state=1).values.tolist()
    elif args.data=='tulu':
        data = load_dataset("allenai/tulu-3-sft-personas-instruction-following")['train'].to_pandas()['prompt'].sample(args.samples, random_state=1).values.tolist()
    elif args.data=='nr':
        nr = load_dataset("HuggingFaceH4/no_robots")["train"].to_pandas()
        nr = nr[nr.category.isin(['Generation', 'Open QA', 'Chat'])]
        data = nr.prompt.sample(args.samples, random_state=1).values.tolist()
    elif args.data=='cnn':        
        data = load_dataset("argilla/cnn-dailymail-summaries")["train"].to_pandas().highlights.sample(args.samples, random_state=1).values.tolist()
    print(hom_bs(data, batch_size=args.batch, verbose=args.noverb))
    
if __name__ == "__main__":
    main()