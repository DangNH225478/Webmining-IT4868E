"""
Course Recommendation Search GUI
Supports both TF-IDF and Ensemble (Deep Learning) search methods
"""

import os
# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel, logging as hf_logging
import warnings
warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()  # Suppress HuggingFace warnings

app = Flask(__name__)

# Global variables for models and data
content_df = None
topic_df = None
tfidf_vectorizer = None
tfidf_content_vectors = None
ensemble_models = None
roberta_tokenizer = None
bert_tokenizer = None
content_vectors_ensemble = None
MODELS_LOADED = False
MAX_LEN = 64

DEVICE = 'cuda'

# ============== Model Definition ==============
class VecModel(nn.Module):
    def __init__(self, model_name, size, has_top=True):
        super(VecModel, self).__init__()
        conf = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_config(conf)
        self.has_top = has_top
        if self.has_top:
            self.bn = nn.BatchNorm1d(size)
            self.top = nn.Linear(size, size)

    def forward(self, ids, mask):
        out = self.backbone(ids, mask)[0]
        out = (out[:, 1:MAX_LEN//2, :] * mask[:, 1:MAX_LEN//2, None]).mean(axis=1)
        if self.has_top:
            out = self.top(self.bn(out))
        return F.normalize(out)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'), strict=False)


def load_data():
    """Load content and topic data"""
    global content_df, topic_df
    
    print("Loading data...")
    content_df = pd.read_csv("data/content.csv")
    topic_df = pd.read_csv("data/topics.csv")
    
    # Preprocess content
    content_df["title"] = content_df["title"].fillna("")
    content_df["description"] = content_df["description"].fillna("")
    content_df["text"] = content_df["text"].fillna("")
    content_df["kind"] = content_df["kind"].fillna("")
    
    # Create combined text field for searching
    content_df["combined_text"] = (
        content_df["title"] + " | " + 
        content_df["kind"] + " | " + 
        content_df["description"].apply(lambda x: str(x)[:256] if pd.notna(x) else "")
    )
    
    # Preprocess topics
    topic_df["title"] = topic_df["title"].fillna("")
    topic_df["description"] = topic_df["description"].fillna("")
    
    print(f"Loaded {len(content_df)} content items and {len(topic_df)} topics")


def initialize_tfidf():
    """Initialize TF-IDF vectorizer and vectors"""
    global tfidf_vectorizer, tfidf_content_vectors
    
    print("Initializing TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="char", 
        ngram_range=(3, 5), 
        min_df=2,
        max_features=50000
    )
    
    # Fit and transform content
    tfidf_content_vectors = tfidf_vectorizer.fit_transform(content_df["combined_text"])
    print("TF-IDF initialization complete")


def initialize_ensemble():
    """Initialize ensemble models (deep learning)"""
    global ensemble_models, roberta_tokenizer, bert_tokenizer, content_vectors_ensemble, MODELS_LOADED
    
    # Check if model files exist
    model_paths = [
        "models/mBERT/Config",
        "models/RoBERTa-Base/Config", 
        "models/RoBERTa-Large/Config",
        "models/mBERT/Weights/mBERT_Full.pth",
        "models/RoBERTa-Base/Weights/RoBERTa_Full.pth",
        "models/RoBERTa-Large/Weights/RoBERTa-Large_Full.pth"
    ]
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Ensemble mode will be disabled.")
            return False
    
    try:
        print("Loading ensemble models (this may take a while)...")
        
        # Load tokenizers
        roberta_tokenizer = AutoTokenizer.from_pretrained("models/RoBERTa-Base/Config")
        bert_tokenizer = AutoTokenizer.from_pretrained("models/mBERT/Config")
        
        # Load models - using CPU due to GPU compatibility
        print(f"Using device: {DEVICE}")
        
        ensemble_models = [
            VecModel("models/mBERT/Config", 768, has_top=False),
            VecModel("models/RoBERTa-Base/Config", 768, has_top=False),
            VecModel("models/RoBERTa-Large/Config", 1024)
        ]
        
        ensemble_models[0].load("models/mBERT/Weights/mBERT_Full.pth")
        ensemble_models[1].load("models/RoBERTa-Base/Weights/RoBERTa_Full.pth")
        ensemble_models[2].load("models/RoBERTa-Large/Weights/RoBERTa-Large_Full.pth")
        
        for model in ensemble_models:
            model.to(DEVICE)
            model.eval()
        
        MODELS_LOADED = True
        print("Ensemble models loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error loading ensemble models: {e}")
        return False


def search_tfidf(query, top_k=20):
    """Search using TF-IDF method"""
    query_vector = tfidf_vectorizer.transform([query])
    
    # Use nearest neighbors for efficient search
    nn_model = NearestNeighbors(n_neighbors=min(top_k, len(content_df)), metric='cosine')
    nn_model.fit(tfidf_content_vectors)
    
    distances, indices = nn_model.kneighbors(query_vector)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        row = content_df.iloc[idx]
        results.append({
            'rank': i + 1,
            'id': row['id'],
            'title': row['title'] if row['title'] else "Untitled",
            'kind': row['kind'] if row['kind'] else "Unknown",
            'description': truncate_text(row['description'], 200),
            'language': row['language'] if pd.notna(row['language']) else "Unknown",
            'similarity': round((1 - dist) * 100, 2)
        })
    
    return results


def search_ensemble(query, top_k=20):
    """Search using Ensemble (deep learning) method with re-ranking"""
    if not MODELS_LOADED:
        return search_tfidf(query, top_k)
    
    model_weights = [0.3, 0.3, 0.4]
    
    # Step 1: Get initial candidates using TF-IDF (get more candidates for re-ranking)
    n_candidates = min(100, len(content_df))
    query_vector = tfidf_vectorizer.transform([query])
    nn_model = NearestNeighbors(n_neighbors=n_candidates, metric='cosine')
    nn_model.fit(tfidf_content_vectors)
    distances, indices = nn_model.kneighbors(query_vector)
    
    # Step 2: Encode query using ensemble models
    with torch.no_grad():
        # RoBERTa encoding for query
        roberta_enc = roberta_tokenizer(
            query, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        roberta_ids = roberta_enc['input_ids'].to(DEVICE)
        roberta_mask = roberta_enc['attention_mask'].to(DEVICE)
        
        # BERT encoding for query
        bert_enc = bert_tokenizer(
            query, 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        bert_ids = bert_enc['input_ids'].to(DEVICE)
        bert_mask = bert_enc['attention_mask'].to(DEVICE)
        
        # Get query embeddings from all models
        query_vec_list = [
            model_weights[0] * ensemble_models[0](bert_ids, bert_mask),
            model_weights[1] * ensemble_models[1](roberta_ids, roberta_mask),
            model_weights[2] * ensemble_models[2](roberta_ids, roberta_mask)
        ]
        query_embedding = torch.cat(query_vec_list, dim=1)
        
        # Step 3: Encode each candidate and compute similarity
        candidates = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            row = content_df.iloc[idx]
            content_text = row['combined_text']
            
            # Encode content
            roberta_enc_c = roberta_tokenizer(
                content_text, 
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            bert_enc_c = bert_tokenizer(
                content_text, 
                padding='max_length', 
                truncation=True, 
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            
            roberta_ids_c = roberta_enc_c['input_ids'].to(DEVICE)
            roberta_mask_c = roberta_enc_c['attention_mask'].to(DEVICE)
            bert_ids_c = bert_enc_c['input_ids'].to(DEVICE)
            bert_mask_c = bert_enc_c['attention_mask'].to(DEVICE)
            
            content_vec_list = [
                model_weights[0] * ensemble_models[0](bert_ids_c, bert_mask_c),
                model_weights[1] * ensemble_models[1](roberta_ids_c, roberta_mask_c),
                model_weights[2] * ensemble_models[2](roberta_ids_c, roberta_mask_c)
            ]
            content_embedding = torch.cat(content_vec_list, dim=1)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(query_embedding, content_embedding).item()
            
            candidates.append({
                'idx': idx,
                'row': row,
                'tfidf_score': 1 - dist,
                'ensemble_score': similarity,
                # Combined score: blend TF-IDF and ensemble
                'combined_score': 0.3 * (1 - dist) + 0.7 * similarity
            })
    
    # Step 4: Sort by combined score (ensemble-weighted)
    candidates.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Step 5: Format results
    results = []
    for i, cand in enumerate(candidates[:top_k]):
        row = cand['row']
        results.append({
            'rank': i + 1,
            'id': row['id'],
            'title': row['title'] if row['title'] else "Untitled",
            'kind': row['kind'] if row['kind'] else "Unknown",
            'description': truncate_text(row['description'], 200),
            'language': row['language'] if pd.notna(row['language']) else "Unknown",
            'similarity': round(cand['combined_score'] * 100, 2)
        })
    
    return results


def truncate_text(text, max_length=200):
    """Truncate text to specified length"""
    if pd.isna(text) or not text:
        return "No description available"
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


@app.route('/')
def index():
    return render_template('index.html', ensemble_available=MODELS_LOADED)


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    method = data.get('method', 'tfidf')
    top_k = data.get('top_k', 20)
    
    if not query:
        return jsonify({'error': 'Please enter a search query', 'results': []})
    
    try:
        if method == 'ensemble' and MODELS_LOADED:
            results = search_ensemble(query, top_k)
        else:
            results = search_tfidf(query, top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'method': method if method == 'tfidf' or MODELS_LOADED else 'tfidf (fallback)',
            'count': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e), 'results': []})


@app.route('/status')
def status():
    return jsonify({
        'tfidf_ready': tfidf_vectorizer is not None,
        'ensemble_ready': MODELS_LOADED,
        'content_count': len(content_df) if content_df is not None else 0,
        'topic_count': len(topic_df) if topic_df is not None else 0
    })


if __name__ == '__main__':
    print("=" * 50)
    print("Course Recommendation Search GUI")
    print("=" * 50)
    
    # Load data
    load_data()
    
    # Initialize TF-IDF (always available)
    initialize_tfidf()
    
    # Try to initialize ensemble (optional)
    initialize_ensemble()
    
    print("\n" + "=" * 50)
    print("Starting server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
