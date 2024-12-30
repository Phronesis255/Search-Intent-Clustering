import pandas as pd
import numpy as np
import py_stringmatching as sm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Load data
serps_input = pd.read_csv('serps_data.csv')

# Filter for top 20 ranks and remove rows with 'url' as None
def filter_twenty_urls(group_df):
    filtered_df = group_df.loc[group_df['url'].notnull()]
    filtered_df = filtered_df.loc[filtered_df['rank'] <= 20]
    return filtered_df

serps_grouped_keyword = serps_input.groupby("keyword")
filtered_serps = serps_grouped_keyword.apply(filter_twenty_urls)
filtered_serps = filtered_serps.reset_index(drop=True)

# Create serp_string by concatenating URLs
def string_serps(df):
    df['serp_string'] = df['url'].str.cat(sep=' ')
    return df

serps_grouped_keyword = filtered_serps.groupby("keyword")
strung_serps = serps_grouped_keyword.apply(string_serps)
strung_serps = strung_serps[['keyword', 'serp_string', 'search_volume']]
strung_serps = strung_serps.drop_duplicates()

# Create keyword pairs without duplicates
keywords = strung_serps['keyword'].unique().tolist()
keyword_pairs = pd.DataFrame([(a, b) for i, a in enumerate(keywords) for b in keywords[i+1:]], columns=['keyword_a', 'keyword_b'])

# Merge to get serp_strings
keyword_pairs = keyword_pairs.merge(strung_serps, left_on='keyword_a', right_on='keyword')
keyword_pairs = keyword_pairs.rename(columns={'serp_string': 'serp_string_a'})
keyword_pairs = keyword_pairs.merge(strung_serps, left_on='keyword_b', right_on='keyword')
keyword_pairs = keyword_pairs.rename(columns={'serp_string': 'serp_string_b'})
keyword_pairs = keyword_pairs[['keyword_a', 'keyword_b', 'serp_string_a', 'serp_string_b']]

# Define similarity function
ws_tok = sm.WhitespaceTokenizer()

def serps_similarity(serps_str1, serps_str2, k=15):
    denom = k + 1
    norm = sum([2 * (1 / i - 1.0 / denom) for i in range(1, denom)])
    serps_1 = ws_tok.tokenize(serps_str1)[:k]
    serps_2 = ws_tok.tokenize(serps_str2)[:k]
    match = lambda a, b: [b.index(x) + 1 if x in b else None for x in a]
    pos_intersections = [(i + 1, j) for i, j in enumerate(match(serps_1, serps_2)) if j is not None]
    pos_in1_not_in2 = [i + 1 for i, j in enumerate(match(serps_1, serps_2)) if j is None]
    pos_in2_not_in1 = [i + 1 for i, j in enumerate(match(serps_2, serps_1)) if j is None]
    a_sum = sum([abs(1 / i - 1 / j) for i, j in pos_intersections])
    b_sum = sum([abs(1 / i - 1 / denom) for i in pos_in1_not_in2])
    c_sum = sum([abs(1 / i - 1 / denom) for i in pos_in2_not_in1])
    intent_prime = a_sum + b_sum + c_sum
    intent_dist = 1 - (intent_prime / norm)
    return intent_dist

# Calculate similarity for each pair
keyword_pairs['similarity'] = keyword_pairs.apply(
    lambda x: serps_similarity(x['serp_string_a'], x['serp_string_b']), axis=1
)

# Cluster keywords based on similarity threshold
simi_lim = 0.4
clusters = {}
cluster_id = 0

for index, row in keyword_pairs.iterrows():
    if row['similarity'] >= simi_lim:
        existing_clusters = [k for k, v in clusters.items() if row['keyword_a'] in v or row['keyword_b'] in v]
        if not existing_clusters:
            clusters[cluster_id] = set([row['keyword_a'], row['keyword_b']])
            cluster_id += 1
        elif len(existing_clusters) == 1:
            clusters[existing_clusters[0]].update([row['keyword_a'], row['keyword_b']])
        else:
            merged_cluster = set().union(*[clusters[k] for k in existing_clusters])
            merged_cluster.update([row['keyword_a'], row['keyword_b']])
            for k in existing_clusters:
                del clusters[k]
            clusters[cluster_id] = merged_cluster
            cluster_id += 1

# Load pre-trained embedding model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Assign cluster names based on highest search volume and calculate average embedding-based similarity
cluster_list = []
for cluster_id, keywords in clusters.items():
    cluster_df = pd.DataFrame({'keyword': list(keywords)})
    cluster_df = cluster_df.merge(strung_serps, on='keyword', how='left')
    cluster_df = cluster_df.sort_values('search_volume', ascending=False)
    cluster_name = cluster_df.iloc[0]['keyword']
    
    # Calculate embeddings for keywords in the cluster
    embeddings = torch.stack([get_embedding(keyword) for keyword in keywords])
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Calculate average similarity score
    avg_similarity = np.tril(similarity_matrix, -1).sum() / (len(keywords) * (len(keywords) - 1) / 2)
    
    cluster_list.append({
        'cluster_id': cluster_id,
        'cluster_name': cluster_name,
        'keywords': list(keywords),
        'avg_similarity': avg_similarity
    })

# Output the clusters
for cluster in cluster_list:
    print(f"Cluster {cluster['cluster_id']} - Named '{cluster['cluster_name']}':")
    print(cluster['keywords'])
    print(f"Average Similarity: {cluster['avg_similarity']:.4f}")
    print()
