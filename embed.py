# -*- coding: utf-8 -*-
"""
Spyder Editor
project/
│
├── whisper_transcribe_output.txt        # output from whisper.cpp
├── quran_embeddings.npy                 # precomputed ayah embeddings
├── ayahs.json                           # ayah text & metadata
└── match_ayah.py                        # main matching script

pip install sentence-transformers numpy faiss-cpu
"""


import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

# Load model (adjust based on your language setup)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Load precomputed ayah embeddings (shape: [N, D])
ayah_embeddings = np.load('quran_embeddings.npy')  # N = number of ayahs

# Load ayah metadata (list of dicts: [{'text': ..., 'surah': ..., 'ayah': ...}, ...])
with open('ayahs.json', 'r', encoding='utf-8') as f:
    ayahs = json.load(f)

# STEP 1: Load partial recitation from whisper.cpp
with open('whisper_transcribe_output.txt', 'r', encoding='utf-8') as f:
    user_recitation = f.read().strip()

print(f"User recited: {user_recitation}")

# STEP 2: Embed user's partial recitation
user_embedding = model.encode(user_recitation, convert_to_tensor=True)

# STEP 3: Compute cosine similarity with each ayah
similarities = util.cos_sim(user_embedding, ayah_embeddings)[0]

# STEP 4: Get top N matches
top_n = 3
top_matches = similarities.topk(k=top_n)

print("\nTop matches:")
for score, idx in zip(top_matches.values, top_matches.indices):
    ayah_info = ayahs[idx]
    print(f"\nScore: {score.item():.4f}")
    print(f"Surah {ayah_info['surah']} - Ayah {ayah_info['ayah']}")
    print(f"Ayah text: {ayah_info['text']}")
