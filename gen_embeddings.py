from sentence_transformers import SentenceTransformer
import sys
from tqdm import tqdm
import ipdb
import numpy as np
import pandas as pd

def main():
    
    # Load the data
    datafile = sys.argv[1]
    df = pd.read_csv(datafile, sep='\t')
    df['response'] = df.response.apply(lambda x: x.strip())
    
    embeddings = None
    
    # Load the model
    model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")
    
    # Encode the responses
    for prompt_id in tqdm(range(100)):
        # Arrange in order persona_id 0 to 99
        responses = df.loc[df.prompt_id==prompt_id].sort_values(by='persona_id')['response'].values.tolist()
        
        passage_embeddings = model.encode(responses)
        
        if embeddings is None:
            embeddings = passage_embeddings
        else:
            embeddings = np.concat((embeddings, passage_embeddings))
        with open(sys.argv[1].split('/')[-2] + '_embeds.npy', 'wb') as f:
            np.save(f, embeddings)    
    
    
if __name__ == "__main__":
    main()