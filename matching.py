from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os
import json
import spacy
from sklearn.metrics.pairwise import cosine_similarity

st = SentenceTransformer('all-MiniLM-L6-v2')

files = []

for root, dirs, filenames in os.walk("./questions_after_filter/"):
    files.extend(filenames)

train = json.load(open('./train.json', 'r'))

print(len(files))


def emb(list_of_sentences):
    embeddings = st.encode(list_of_sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

for i, f in enumerate(files):

    print(i)
    image_id = f.split('.')[0]
    
    original_questions = []
    for j in train:
        if j['img_id'] == str(image_id):
            original_questions.append(j['sent'])

    #print(original_questions)

    #print()
    questions_after_filter = pickle.load(open(f"./questions_after_filter/{f}", "rb"))
    #print(questions_after_filter)

    question2cosine_sim = {}
    for x, y in questions_after_filter.items():
        cosine_sim = [cosine_similarity([emb(x)], [emb(i)]).tolist()[0][0] for i in original_questions]
        question2cosine_sim[(x, tuple(y))] = cosine_sim


    questions_after_matching = {}
    for k, v in question2cosine_sim.items():
        if sum(list(map(lambda x: x < 0.5, v))) > 2:
            questions_after_matching[k[0]] = set(k[1])


    #print()
    #print(len(questions_after_matching))

    pickle.dump(questions_after_matching, open(f"./questions_after_matching/{f}", "wb"))


