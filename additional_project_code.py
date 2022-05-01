from collections import defaultdict
import json
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pyspark
from pyspark.mllib.feature import HashingTF
from tqdm import tqdm
from wordcloud import WordCloud


######## Section 1 ########


def setup_nltk():
    """
    Downloads nltk resources
    """
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

def lines_generator(n=float('inf'), filename='./arxiv-metadata-oai-snapshot.json'):
    """
    Returns iterator of n objects, each one json decoded from a line of file filename
    """
    with open(filename) as f:
        for i, line in enumerate(f):
            if i >= n: break
            try:
                yield json.loads(line)
            except: continue


######## Section 3 ########

def unique_token_counts(rdd: pyspark.RDD, column: str) -> pyspark.RDD:
    return rdd\
        .map(lambda row: row[column].split(' '))\
        .flatMap(lambda x:x)\
        .map(lambda x:[x])\
        .countByKey()

def create_wordcloud(token_values: dict):
    wordcloud = WordCloud().generate_from_frequencies(token_values)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def get_hashes_dict(rdd: pyspark.RDD, hashingTF: HashingTF, column: str) -> defaultdict:
    unique_tokens = rdd\
        .map(lambda row: row[column].split(' '))\
        .flatMap(lambda x:x)\
        .map(lambda x:[x])\
        .cache()
    
    hashes = hashingTF\
        .transform(unique_tokens)\
        .map(lambda vector: vector.indices[0])\
    
    hashed_tokens = hashes.zip(unique_tokens.map(lambda x: x[0]))
    n_tokens = unique_tokens.count()
    token_counts = unique_tokens.countByKey()

    hashes = defaultdict(lambda: (str(), 0))
    for hash_val, token in tqdm(hashed_tokens.toLocalIterator(), total=n_tokens):
        count = token_counts.get(token, 0)
        hashes[hash_val] = max((token, count), hashes[hash_val], key=lambda pair: pair[1])
        
    return hashes

    
######## Section 4 ########

def resparsify(sv: pyspark.mllib.linalg.SparseVector, unique_hash_index) -> pyspark.mllib.linalg.SparseVector:
    nonzero_inds = sv.values != 0
    new_indices = [unique_hash_index[i] for i in sv.indices[nonzero_inds]]
    
    return pyspark.mllib.linalg.SparseVector(len(unique_hash_index), zip(new_indices, sv.values[nonzero_inds]))

def combine(defaultdicts):
    result = defaultdict(int)
    for d in defaultdicts:
        for k, v in d.items():
            result[k] += v
    return result

def dict_to_numpy(dictionary, n):
    values = np.zeros((n, n), dtype=np.int32)
    for (i, j), value in dictionary.items():
        values[int(i), int(j)] = int(value)
        
    return values


