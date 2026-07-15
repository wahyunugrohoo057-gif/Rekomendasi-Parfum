from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ── Load Dataset ───────────────────────────────────────────────────────────────
df = pd.read_csv('Parfume Lokal Indonesian.csv')
df.columns = df.columns.str.strip()

# Normalize kolom names
rename_map = {}
for col in df.columns:
    low = col.lower().strip()
    if low in ['top notes', 'top_notes']:
        rename_map[col] = 'top_notes'
    elif low in ['mid notes', 'mid_notes', 'middle notes']:
        rename_map[col] = 'mid_notes'
    elif low in ['base notes', 'base_notes']:
        rename_map[col] = 'base_notes'
df = df.rename(columns=rename_map)

# Pastikan kolom wajib ada
for col in ['top_notes', 'mid_notes', 'base_notes']:
    if col not in df.columns:
        df[col] = ''

df = df.drop_duplicates(subset=['perfume'])
df = df.fillna('')
df = df.reset_index(drop=True)

print(f"✓ Dataset loaded: {len(df)} parfum")
print(f"  Kolom: {df.columns.tolist()}")

# ── Preprocessing ──────────────────────────────────────────────────────────────
STOPWORDS = {
    'dan','yang','pada','dengan','untuk','dari','dalam','ini','itu','atau',
    'juga','adalah','di','ke','the','a','an','and','of','with','in','for',
    'note','notes','top','mid','base','middle'
}

def preprocess(text):
    if not text or pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t.strip() for t in text.split() if t.strip() and t.strip() not in STOPWORDS]
    return ' '.join(tokens)

# Gabungkan top + mid + base notes (sesuai paper)
df['aroma_combined'] = (
    df['top_notes'].apply(preprocess) + ' ' +
    df['mid_notes'].apply(preprocess) + ' ' +
    df['base_notes'].apply(preprocess)
).str.strip()

print(f"  Contoh aroma_combined[0]: {df['aroma_combined'].iloc[0][:80]}")

# ── TF-IDF Vectorization ───────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    min_df=1,
    max_df=0.95,
    ngram_range=(1, 2),   # unigram + bigram sesuai paper
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(df['aroma_combined'])
print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

# ── Cosine Similarity Matrix ───────────────────────────────────────────────────
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"  Cosine sim matrix shape: {cosine_sim.shape}")

# ── Fungsi Rekomendasi ─────────────────────────────────────────────────────────
def get_recommendations(perfume_name, top_n=5):
    """
    Content-Based Filtering:
    1. Cari index parfum query
    2. Ambil similarity scores semua parfum terhadap query
    3. Sort descending
    4. Return Top-N (exclude parfum itu sendiri)
    """
    # Case-insensitive match
    mask = df['perfume'].str.lower().str.strip() == perfume_name.lower().strip()
    matches = df[mask]

    if matches.empty:
        print(f"  ✗ Parfum tidak ditemukan: '{perfume_name}'")
        return [], {}

    idx = matches.index[0]
    print(f"  ✓ Parfum ditemukan di index {idx}: {df.iloc[idx]['perfume']}")

    # Ambil semua similarity scores untuk parfum ini
    sim_scores = list(enumerate(cosine_sim[idx]))
    print(f"  Total similarity scores: {len(sim_scores)}")

    # Sort descending by score, exclude diri sendiri
    sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores_filtered = [(i, s) for i, s in sim_scores_sorted if i != idx]

    print(f"  Top-5 similarity: {[(df.iloc[i]['perfume'][:20], round(s,4)) for i,s in sim_scores_filtered[:5]]}")

    # Ambil Top-N
    top_results = sim_scores_filtered[:top_n]

    results = []
    for rank, (i, score) in enumerate(top_results, 1):
        r = df.iloc[i]
        results.append({
            'rank': rank,
            'id': str(r.get('ID_Perfume', '')),
            'name': str(r['perfume']),
            'brand': str(r.get('brand', '-')),
            'top_notes': str(r['top_notes']),
            'mid_notes': str(r['mid_notes']),
            'base_notes': str(r['base_notes']),
            'situation': str(r.get('situation', '-')),
            'gender': str(r.get('gender', '-')),
            'price': str(r.get('price', '-')),
            'similarity_score': round(float(score), 4),
            'similarity_pct': round(float(score) * 100, 2),
        })

    # Info parfum query
    q = df.iloc[idx]
    query_info = {
        'name': str(q['perfume']),
        'brand': str(q.get('brand', '-')),
        'top_notes': str(q['top_notes']),
        'mid_notes': str(q['mid_notes']),
        'base_notes': str(q['base_notes']),
        'situation': str(q.get('situation', '-')),
        'gender': str(q.get('gender', '-')),
        'price': str(q.get('price', '-')),
    }

    return results, query_info

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    cols = ['perfume', 'brand', 'top_notes', 'mid_notes', 'base_notes', 'gender', 'situation', 'price']
    # tambahkan ID_Perfume kalau ada
    if 'ID_Perfume' in df.columns:
        cols = ['ID_Perfume'] + cols

    perfumes = df[[c for c in cols if c in df.columns]].copy()
    perfumes = perfumes.fillna('-')
    return render_template('index.html',
                           perfumes=perfumes.to_dict(orient='records'),
                           total=len(df))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data', 'recommendations': [], 'query': {}}), 400

    perfume_name = data.get('perfume_name', '').strip()
    top_n = int(data.get('top_n', 5))

    print(f"\n── /recommend ──")
    print(f"  perfume_name: '{perfume_name}'")
    print(f"  top_n: {top_n}")

    if not perfume_name:
        return jsonify({'error': 'perfume_name kosong', 'recommendations': [], 'query': {}}), 400

    results, query_info = get_recommendations(perfume_name, top_n)

    print(f"  Jumlah hasil: {len(results)}")

    return jsonify({
        'query': query_info,
        'recommendations': results,
        'total': len(results)
    })

@app.route('/search')
def search():
    q = request.args.get('q', '').strip().lower()
    if not q:
        return jsonify([])
    mask = (
        df['perfume'].str.lower().str.contains(q, na=False) |
        df['brand'].str.lower().str.contains(q, na=False)
    )
    res = df[mask][['perfume', 'brand']].head(10).fillna('-')
    return jsonify(res.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)