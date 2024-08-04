import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import requests
import plotly.express as px

st.set_page_config(layout='wide', page_title='Book Recommender', page_icon='book')
# Load the model

@st.cache_data
def get_data():
    meta = pd.read_csv('Books_df.csv')
    user = pd.read_csv('Output_csv.csv')
    return meta, user

meta, user = get_data()

home_tab, graph_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Grafikler","Öneri Sistemi"])

 # Verileri işleyin ve ölçekleyin
scaler = MinMaxScaler()
user_scaled = scaler.fit_transform(user)

    # NaN değerleri kaldırın
nan_mask = np.isnan(user_scaled).any(axis=1)
user_scaled = user_scaled[~nan_mask]

    # K-Means kümeleme
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(user_scaled)

    # PCA ile boyut indirgeme
pca = PCA(n_components=2)
user_pca = pca.fit_transform(user_scaled)

def find_similar_books(book_title, meta, user, top_n=5, genre=None, sub_genre=None):
    # Filtreleme işlemleri
    if genre:
        genre_df = meta[meta['Main Genre'] == genre]
        genre_indices = genre_df.index
        user_pca = user_pca[genre_indices]
    else:
        genre_df = meta
    
    if sub_genre:
        genre_df = genre_df[genre_df['Sub Genre'] == sub_genre]
        genre_indices = genre_df.index
        user_pca = user_pca[genre_indices]

    # Kısmi eşleşmeyi bul ve en yakın başlığı seç
    titles = genre_df['Title'].tolist()
    best_match = process.extractOne(book_title.strip(), titles)
    
    if best_match is None or best_match[1] < 80:  # Benzerlik skoru eşiği
        raise ValueError(f"'{book_title}' kitabı veri setinde bulunmuyor.")

    matched_title = best_match[0]
    idx = genre_df[genre_df['Title'] == matched_title].index[0]
    
    # Cosine benzerlik hesapla
    cosine_sim = cosine_similarity(user_pca)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    
    book_indices = [i[0] for i in sim_scores]
    similar_books = genre_df.iloc[book_indices]
    
     # İlk kitabı önerilenlerden çıkararak tekrarını engelleme
    book_indices = [i[0] for i in sim_scores if genre_df.iloc[i[0]]['Title'] != matched_title][:top_n]
    similar_books = genre_df.iloc[book_indices]
    
    # Tekrar eden başlıkları kaldır
    unique_books = similar_books.drop_duplicates(subset='Title', keep='first')
    
    # Alt türlere göre filtreleme
    filtered_books = unique_books.groupby('Sub Genre').first().reset_index()
      
    return filtered_books




