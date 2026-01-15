import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Sub in for the actual corpus of past queries
# size is assumed to average person's search history over past 4 days (4 searches per day)
corpus = [
    "where to get chocolates",
    "nearest bakery for chocolates",
    "Python bootcamp online",
    "learn Python from home",
    "nearest Mc Donald's",
    "nearest fast food restaurant",
    "Cute kitten Videos",
    "Funny cat videos",
    "How to fix slow Wi-Fi",
    "Who won the Oscar for Best Actor 2025",
    "Recipes for chicken curry",
    "Meaning of life quotes",
    "Local gym membership prices",
    "How to exercise for beginners",
    "History of the Great Wall of China",
    "History of the Pyramids in Egypt",
]



    
# function to compute cosine similarity and filter results
def cosine_similarity_filter(model,query, corpus):
    # compute cosine similarity
    q_vec = model.encode(query, normalize_embeddings=True)
    d_vecs = model.encode(corpus, normalize_embeddings=True)
    scores = np.dot(d_vecs, q_vec)  # shape [num_docs]
    # filter results with similarity score >= 0.75
    filtered = filter(lambda x: x[0] >= 0.75, zip(scores.tolist(), corpus))
    # return only the texts from filtered results
    return [f for _, f in filtered]

# function to perform agglomerative clustering on the remaining queries
def Agglomerative_Clustering(corpus, model, distance_threshold=0.7):
    # encodes corpus into unit vectors
    X = model.encode(corpus, normalize_embeddings=True)
    # creates an agglomerative clustering model
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    # perform agglomerative clustering
    clustering.fit(X)
    # create clusters based on labels
    # labels is a list of cluster assignments for each input text
    # each key is a cluster label, value is list of indices in that cluster
    clusters = {}
    for index, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(index)
    
    # prepare the output matrix
    matrix = []
    for key in range(len(clusters)):
        for label, items in clusters.items():
            # checks if the label is equal to the current key aka matrix[0] is cluster 0
            if key == label:
                texts = []
                # creates new list with texts from corpus based on the indices in items
                for text in items:
                    texts.append(corpus[text])
                # appends the texts into the matrix
                matrix.append(texts)

    return matrix

# sees which cluster has highest average cosine similarity
def cos_sim_avg (model,query, matrix):
    avg_scores = []
    for cluster in range(len(matrix)):
        corpus = matrix[cluster]
        # keeps track of scores for each cluster
        scores = []
        for text in range(len(corpus)):
            # calculates cosine similarity for each text in cluster
            text = corpus[text]
            t_vec = model.encode(text, normalize_embeddings=True)
            q_vec = model.encode(query, normalize_embeddings=True)
            score = np.dot(t_vec, q_vec)
            scores.append(score)
        # calculates average score for each cluster and appends to list
        avg_score = np.mean(scores)
        avg_scores.append(avg_score)
    return avg_scores

def main(query,model):
    #filter out corpus based on cosine similarity (score >= 0.75)
    results = cosine_similarity_filter(model, query, corpus)
    if not results:
        return None
    # perform agglomerative clustering on remaining queries based on cosine similarity
    clusters = Agglomerative_Clustering(results, model)
    # get average cosine similarity for each cluster
    avg_scores = cos_sim_avg(model, query, clusters)
    # find the cluster with highest average cosine similarity
    max_index = np.argmax(avg_scores)
    docs = clusters[max_index]
    # if more than 5 docs in the selected cluster, select top 5 based on cosine similarity
    if len(docs) >= 5:
        # get cosine similarity scores for text in the selected cluster
        q_vec = model.encode(query, normalize_embeddings=True)
        d_vecs = model.encode(docs, normalize_embeddings=True)
        scores = np.dot(d_vecs, q_vec)  # shape [num_docs]  
        # rank and select top 5 documents
        ranked = sorted(zip(scores.tolist(), docs), key=lambda x: x[0], reverse=True)
        docs = [r for _, r in ranked[:5]]   
    
    return docs
