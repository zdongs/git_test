import pickle

with open("data/corpus-2014.data",'rb') as f:
    corpus = pickle.load(f)

print(corpus[42])