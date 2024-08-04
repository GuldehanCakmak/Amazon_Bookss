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



def get_data():
    meta = pd.read_csv('data/streamlit_Books_df.csv')
    user = pd.read_csv('data/streamlit_Output_csv.csv')
    return meta, user

meta, user = get_data()

# Kitap önerisi fonksiyonu
def find_similar_books(book_title, meta, user, top_n=5, genre=None, sub_genre=None):
    # Filtreleme işlemleri
    if genre:
        genre_df = meta[meta['Main Genre'] == genre]
        genre_indices = genre_df.index
        user = user[genre_indices]
    else:
        genre_df = meta
    
    if sub_genre:
        genre_df = meta[meta['Sub Genre'] == sub_genre]
        genre_indices = genre_df.index
        user = user[genre_indices]

    # Kısmi eşleşmeyi bul ve en yakın başlığı seç
    titles = genre_df['Title'].tolist()
    best_match = process.extractOne(book_title.strip(), titles)
    
    if best_match is None or best_match[1] < 80:  # Benzerlik skoru eşiği
        st.error(f"'{book_title}' kitabı veri setinde bulunmuyor.")
        return pd.DataFrame()

    matched_title = best_match[0]
    idx = genre_df[genre_df['Title'] == matched_title].index[0]
    
    # Cosine benzerlik hesapla
    cosine_sim = cosine_similarity(features_pca)
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

# Hava durumu verisini almak için OpenWeatherMap API'si
def get_weather(api_key, city):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()
    return data

# Hava durumuna göre kitap önerileri
def suggest_books_by_weather(weather_condition, book_format='text'):
    recommendations = {
        'Clear': {
            'text': ['The Alchemist by Paulo Coelho', 'To Kill a Mockingbird by Harper Lee'],
            'audio': ['Becoming by Michelle Obama (Audiobook)', 'Educated by Tara Westover (Audiobook)']
        },
        'Rain': {
            'text': ['The Girl with the Dragon Tattoo by Stieg Larsson', 'Gone Girl by Gillian Flynn'],
            'audio': ['The Silent Patient by Alex Michaelides (Audiobook)', 'Big Little Lies by Liane Moriarty (Audiobook)']
        },
        'Snow': {
            'text': ['Harry Potter Series by J.K. Rowling', 'The Hobbit by J.R.R. Tolkien'],
            'audio': ['A Game of Thrones by George R.R. Martin (Audiobook)', 'The Lion, the Witch and the Wardrobe by C.S. Lewis (Audiobook)']
        },
        'Clouds': {
            'text': ['The Catcher in the Rye by J.D. Salinger', '1984 by George Orwell'],
            'audio': ['The Great Gatsby by F. Scott Fitzgerald (Audiobook)', 'The Handmaid\'s Tale by Margaret Atwood (Audiobook)']
        }
    }
    return recommendations.get(weather_condition, {'text': ['Pride and Prejudice by Jane Austen', 'Moby Dick by Herman Melville'],
                                                   'audio': ['The Odyssey by Homer (Audiobook)', 'Jane Eyre by Charlotte Bronte (Audiobook)']}).get(book_format)

# Streamlit uygulaması
st.title('Kitap Tavsiye Sistemi')

# Kitap veri setini yükleyin
uploaded_file = st.file_uploader("CSV dosyasını yükleyin", type="csv")
if uploaded_file is not None:
    meta = pd.read_csv(uploaded_file)
    st.write("Veri seti başarıyla yüklendi.")
    
    # Verileri işleyin ve ölçekleyin
    user = meta[['Price', 'Rating', 'No. of People Rated', 'Main Genre']]  # Özellik sütunlarını belirleyin
    scaler = MinMaxScaler()
    user_scaled = scaler.fit_transform(features)

    # NaN değerleri kaldırın
    nan_mask = np.isnan(user_scaled).any(axis=1)
    user_scaled = features_scaled[~nan_mask]

    # K-Means kümeleme
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(user_scaled)

    # PCA ile boyut indirgeme
    pca = PCA(n_components=2)
    user_pca = pca.fit_transform(user_scaled)

    # Kitap tavsiyesi
    book_title = st.text_input('Kitap adı girin:')
    if book_title:
        similar_books = find_similar_books(book_title, df, features_pca)
        if not similar_books.empty:
            st.write(f"'{book_title}' kitabını alan kullanıcıya önerilen kitaplar:")
            st.write(similar_books[['Title', 'Author', 'Main Genre', 'Sub Genre']])

# Hava durumuna göre kitap önerisi
api_key = st.text_input('OpenWeatherMap API anahtarınızı girin:')
city = st.text_input('Şehir adı girin:')
if api_key and city:
    weather_data = get_weather(api_key, city)
    if 'weather' in weather_data:
        weather_condition = weather_data['weather'][0]['main']
        book_format = st.selectbox('Kitap formatını seçin:', ['text', 'audio'])
        book_recommendations = suggest_books_by_weather(weather_condition, book_format)
        st.write(f"Hava durumu: {weather_condition}")
        st.write("Önerilen Kitaplar:")
        for book in book_recommendations:
            st.write(f"- {book}")
    else:
        st.write("Hava durumu bilgisi alınamadı. API yanıtını kontrol edin.")

st.title(':blue[Miuul] Movie :blue[Recommender] 🎥', )

home_tab, graph_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Grafikler","Öneri Sistemi"])


# ADAMIN KODLARI

# home tab

col1, col2, col3 = home_tab.columns([1,1,1])
col1.image("https://www.looper.com/img/gallery/star-wars-how-darth-vaders-costume-limited-the-duel-in-a-new-hope/l-intro-1683252662.jpg")
col1.subheader("Nedir?")
col1.markdown('*Film dünyası geniş bir deniz gibi; her türden, her dilden ve her duygudan eserlerle dolu. Bizim film öneri sistemi, size tam da bu denizde yol gösterecek. Sizin ilgi alanlarınıza, beğenilerinize ve tercihlerinize göre özenle seçilmiş filmleri öneriyoruz. Üstelik, algoritma her geçen gün sizinle daha iyi anlaşacak ve beğenilerinizi daha doğru tahmin edecek şekilde gelişiyor.*')
col1.audio("http://soundfxcenter.com/movies/star-wars/8d82b5_Star_Wars_The_Imperial_March_Theme_Song.mp3")

col2.subheader("Nasıl çalışır?")
col2.markdown("Sistemimiz, karmaşık bir yapay zeka algoritmasıyla çalışır. İlk önce sizden bazı tercihlerinizi ve beğenilerinizi belirlememizi isteriz. Sonra, bu bilgileri kullanarak, benzer kullanıcıların beğenilerine göre filmleri öneririz. Ayrıca, izlediğiniz filmlere göre sistemi güncelleyerek size daha kişiselleştirilmiş öneriler sunarız. Böylece her ziyaretinizde yeni ve ilginizi çekebilecek filmler keşfedebilirsiniz.")
col2.image("https://media.vanityfair.com/photos/5e2871cdb8e7e70008021290/master/pass/M8DBESU_EC004.jpg")

col3.image("https://media3.giphy.com/media/spu2k869TI1aw/giphy.gif?cid=6c09b952gtje6mb1utxznqgjzphn2afpoh1105w4czl89oxw&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=g")
col3.subheader("Ne işe yarar?")
col3.markdown("Film öneri sistemi ile maceraya hazır mısınız? Sizi keşfetmek istediğiniz türlerde, heyecan verici ve unutulmaz filmlerle buluşturmak için buradayız. Sadece birkaç adımda, film dünyasının en iyilerini keşfetmek ve favorilerinizi bulmak mümkün olacak. Üstelik, sistemimiz sürekli olarak güncellenir ve sizin tercihlerinize göre daha iyi hale gelir. Yeni filmler keşfetmek ve sinema deneyiminizi zenginleştirmek için hemen şimdi başlayın!")


# graph tab

fig = px.bar(data_frame=meta.sort_values(by="revenue", ascending=False).head(10),
                 x="revenue",
                 y="original_title",
                 orientation="h",
                 hover_data=["release_date"],
                 color="vote_average",
                 color_continuous_scale='blues')
                 
graph_tab.plotly_chart(fig)

genres = ['Title', 'Author', 'Main Genre', 'Rating', 'No. of People rated']
selected_genre = graph_tab.selectbox(label="Tür seçiniz", options=genres)
graph_tab.markdown(f"Seçilen tür: **{selected_genre}**")

graph_tab.dataframe(meta.loc[meta.genres_x.str.contains(selected_genre), ['title', 'genres_x', 'release_date', 'vote_average']].sort_values(by="vote_average", ascending=False).head(10))

# recommendation_tab

r_col1, r_col2, r_col3 = recommendation_tab.columns([1,2,1])
selected_movie = r_col2.selectbox("Kitap seçiniz.", options=meta.title.unique())
recommendations = user.corrwith(user[selected_movie]).sort_values(ascending=False)[1:6]

movie_one, movie_two, movie_three, movie_four, movie_five = recommendation_tab.columns(5)

recommend_button = r_col2.button("Film Öner")

if recommend_button:
        for index, movie_col in enumerate([movie_one, movie_two, movie_three, movie_four, movie_five]):
            movie = meta.loc[meta.title == recommendations.index[index], :]
            movie_col.subheader(f"**{movie.title.values[0]}**")
            movie_col.image(get_image_from_imdb(movie.imdb_id.values[0]))
            movie_col.markdown(f"**{movie.vote_average.values[0]}**")


