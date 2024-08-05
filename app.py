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

# Yazardaki boşlukları Unknown ile doldur
meta['Author'].fillna('Unknown', inplace=True)
meta = meta[(meta['Rating'] != 0) & (meta['Price'] != 0)]
# Price daki para birimi işaretini kaldır
meta['Price'] = meta['Price'].str.replace('₹', '').str.replace(',', '').astype(float)



# home tab
home_tab, graph_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Grafikler","Öneri Sistemi"])
col1, col2, col3 = home_tab.columns([1,1,1])
col1.image("https://st4.depositphotos.com/1544309/21848/i/1600/depositphotos_218482086-stock-photo-tasty-cup-coffee-flowers-books.jpg")
col1.subheader("Nedir?")
col1.markdown("*Merhaba sevgili kitap severler! Ben bir kitap kurdu olarak her zaman yeni ve ilginç kitaplar keşfetmeyi, okumayı ve bu kitapları arkadaşlarımla paylaşmayı çok severim. Geçenlerde, Amazon'un devasa kitap veri tabanını keşfetmeye karar verdim. Amacım, arkadaşlarıma onların zevklerine en uygun kitapları önermek ve bu devasa bilgi denizinden en iyi şekilde faydalanmak oldu. İşte bu serüvenin hikayesi ve sonuçları!*")
col1.audio("http://soundfxcenter.com/movies/star-wars/8d82b5_Star_Wars_The_Imperial_March_Theme_Song.mp3")

col2.subheader("Hangi Kitap?")
col2.markdown("Verileri analiz etmeye başladığımda, bazı kitapların binlerce kez değerlendirildiğini ve yüksek puanlar aldığını fark ettim. Diğer kitaplar ise daha az ilgi görmüştü ama belirli bir okuyucu kitlesi tarafından çok beğenilmişti. Bu bilgiler ışığında, arkadaşlarımın okuma alışkanlıklarına göre özelleştirilmiş öneriler sunabileceğimi anladım.")
col2.image("https://media.vanityfair.com/photos/5e2871cdb8e7e70008021290/master/pass/M8DBESU_EC004.jpg")

col3.image("https://media3.giphy.com/media/spu2k869TI1aw/giphy.gif?cid=6c09b952gtje6mb1utxznqgjzphn2afpoh1105w4czl89oxw&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=g")
col3.subheader("Sizin icin burdayim")
col3.markdown("*Hava Durumu ile Kitap Önerisi: Hava durumu verilerini alarak her gün için uygun kitap önerileri sunan bir sistem geliştirdim. Örneğin, yağmurlu bir gün için sıcak bir kahve eşliğinde okunabilecek bir klasik roman öneriyorum. Güneşli ve neşeli günler için ise enerjinizi artıracak macera kitapları öneriyorum.*")
col3.markdown("*Sesli Kitap Önerileri: Sesli kitapları seven arkadaşlarım için de öneriler sundum. Yürüyüş yaparken, araba kullanırken veya rahatlamak istediğiniz anlarda dinleyebileceğiniz en iyi sesli kitapları seçtim.*")
col3.markdown("*Sonuçlar gerçekten de heyecan vericiydi! Arkadaşlarıma kişiselleştirilmiş kitap önerilerinde bulundum. Onlara şunları söyledim:*") 
col3.markdown("*Ahmet, sen polisiye romanları çok seviyorsun. İşte bu yağmurlu gün için mükemmel bir öneri:  Sherlock Holmes serisi. Eminim ki seni çok heyecanlandıracak!*")
col3.markdown("*Ayşe, senin için harika bir romantik kitap buldum. Hava güneşli ve senin de keyfin yerinde.  Pride and Prejudice tam sana göre!*")
col3.markdown("*Mehmet, sesli kitapları sevdiğini biliyorum. İşte işe giderken dinleyebileceğin bir kitap:   Sapiens: İnsanlığın Kısa Tarihi . Eminim çok şey öğreneceksin.*")


# graph tab
fig = px.bar(data_frame=meta.sort_values(by="Main Genre", ascending=False).head(10),
                 x="Main Genre",
                 y="Title",
                 orientation="h",
                 hover_data=["Rating"],
                 color="Rating",
                 color_continuous_scale='blues')
                 
graph_tab.plotly_chart(fig)

genres = ["Arts, Film & Photography", "Children's Books", "Fantasy, Horror & Science Fiction", "Comics & Mangas", "Romance"]
selected_genre = graph_tab.selectbox(label="Tür seçiniz", options=genres)
graph_tab.markdown(f"Seçilen tür: **{selected_genre}**")

graph_tab.dataframe(meta.loc[meta.genres_x.str.contains(selected_genre), ['Title', 'Main Genre', 'Rating']].sort_values(by="Rating", ascending=False).head(10))

# recommendation_tab
r_col1, r_col2, r_col3 = recommendation_tab.columns([1,2,1])


def find_similar_books(book_title, meta, user_pca, top_n=5, genre=None, sub_genre=None):
    # Filtreleme işlemleri
    if genre and genre != 'None':
        meta = meta[meta['Main Genre'] == genre]
        genre_indices = meta.index
        user_pca = user_pca[genre_indices]
        
    if sub_genre and sub_genre != 'None':
        meta = meta[meta['Sub Genre'] == sub_genre]
        genre_indices = meta.index
        user_pca = user_pca[genre_indices]

    # Kısmi eşleşmeyi bul ve en yakın başlığı seç
    titles = meta['Title'].tolist()
    best_match = process.extractOne(book_title.strip(), titles)
    
    if best_match is None or best_match[1] < 80:  # Benzerlik skoru eşiği
        raise ValueError(f"'{book_title}' kitabı veri setinde bulunmuyor.")

    matched_title = best_match[0]
    idx = meta[meta['Title'] == matched_title].index[0]
    
    # Cosine benzerlik hesapla
    cosine_sim = cosine_similarity(user_pca)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    
    book_indices = [i[0] for i in sim_scores]
    similar_books = meta.iloc[book_indices]
    
    # Tekrar eden başlıkları kaldır
    unique_books = similar_books.drop_duplicates(subset='Title', keep='first')
    
    return unique_books

# Streamlit uygulaması
st.title('Kitap Tavsiye Sistemi')

# Kullanıcıdan girdi alma
selected_book = st.text_input("Kitap Başlığını Girin:")


# Dinamik tür güncellemeleri
if 'selected_book' in st.session_state:
    # Seçili kitaba göre türlerin güncellenmesi
    filtered_meta = meta[meta['Title'] == st.session_state.selected_book]
    if not filtered_meta.empty:
        unique_genres = list(filtered_meta['Main Genre'].unique())
        unique_sub_genres = list(filtered_meta['Sub Genre'].unique())
    else:
        unique_genres = list(meta['Main Genre'].unique())
        unique_sub_genres = list(meta['Sub Genre'].unique())
else:
    unique_genres = list(meta['Main Genre'].unique())
    unique_sub_genres = list(meta['Sub Genre'].unique())

# Ana tür ve alt tür seçimleri
selected_genre = st.selectbox("Ana Tür Seçin:", options=unique_genres + ['None'])
selected_sub_genre = st.selectbox("Alt Tür Seçin:", options=unique_sub_genres + ['None'])

# Tavsiye butonu
if st.button('Kitap Tavsiye Et'):
    if not selected_book:
        st.warning("Lütfen bir kitap başlığı girin.")
    else:
        st.session_state.selected_book = selected_book  # Kitap başlığını session state'e kaydet
        try:
            recommended_books = find_similar_books(
                selected_book,
                meta,
                user_pca,
                genre=None if selected_genre == 'None' else selected_genre,
                sub_genre=None if selected_sub_genre == 'None' else selected_sub_genre
            )
            if recommended_books.empty:
                st.write("Maalesef öneri bulunamadı.")
            else:
                st.write("Önerilen Kitaplar:")
                for index, row in recommended_books.iterrows():
                    try:
                        st.image(row['URLs'], caption=row['Title'], use_column_width=True)
                    except Exception as e:
                        st.warning(f"Resim yüklenirken bir hata oluştu: {e}")
        except ValueError as e:
            st.error(e)
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")


