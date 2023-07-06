import sys
import pickle
from tqdm import tqdm

token_dist = {}

i = sys.argv[1]
corpus_name = sys.argv[2]
print(i)

for file in tqdm(pickle.load(open('{}_file_{}'.format(corpus_name, i),'rb'))):
    text = pickle.load(open(file,'rb'))
    
    for doc in text:
        for sent in doc:
            for token in sent:
                token_dist[token] = token_dist.get(token,0) + 1
                
                
pickle.dump(token_dist, open('{}_token_dist_{}'.format(corpus_name, i),'wb'))