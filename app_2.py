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
    top_books = pd.read_csv('top_books_per_genre.csv')
    return meta, user, top_books

meta, user, top_books = get_data()

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
col1, col2, col3 = home_tab.columns([2,2,1])
col1.image("https://i.pinimg.com/564x/e8/f1/37/e8f13739ab419520d532e0a426c76298.jpg")
col1.subheader("Nedir?")
col1.markdown("*Merhaba sevgili kitap severler! Ben bir kitap kurdu olarak her zaman yeni ve ilginç kitaplar keşfetmeyi, okumayı ve bu kitapları arkadaşlarımla paylaşmayı çok severim. Geçenlerde, Amazon'un devasa kitap veri tabanını keşfetmeye karar verdim. Amacım, arkadaşlarıma onların zevklerine en uygun kitapları önermek ve bu devasa bilgi denizinden en iyi şekilde faydalanmak oldu. İşte bu serüvenin hikayesi ve sonuçları!*")
col1.markdown("*Verileri analiz etmeye başladığımda, bazı kitapların binlerce kez değerlendirildiğini ve yüksek puanlar aldığını fark ettim. Diğer kitaplar ise daha az ilgi görmüştü ama belirli bir okuyucu kitlesi tarafından çok beğenilmişti. Bu bilgiler ışığında, arkadaşlarımın okuma alışkanlıklarına göre özelleştirilmiş öneriler sunabileceğimi anladım.*")

col2.subheader("Hangi Kitap?")
col2.markdown("*Hava Durumu ile Kitap Önerisi: Hava durumu verilerini alarak her gün için uygun kitap önerileri sunan bir sistem geliştirdim. Örneğin, yağmurlu bir gün için sıcak bir kahve eşliğinde okunabilecek bir klasik roman öneriyorum. Güneşli ve neşeli günler için ise enerjinizi artıracak macera kitapları öneriyorum.*")
col2.markdown("*Sesli Kitap Önerileri: Sesli kitapları seven arkadaşlarım için de öneriler sundum. Yürüyüş yaparken, araba kullanırken veya rahatlamak istediğiniz anlarda dinleyebileceğiniz en iyi sesli kitapları seçtim.*")
col2.markdown("*Sonuçlar gerçekten de heyecan vericiydi! Arkadaşlarıma kişiselleştirilmiş kitap önerilerinde bulundum. Onlara şunları söyledim:*") 
col2.markdown("*Ahmet, sen polisiye romanları çok seviyorsun. İşte bu yağmurlu gün için mükemmel bir öneri:  Sherlock Holmes serisi. Eminim ki seni çok heyecanlandıracak!*")
col2.markdown("*Ayşe, senin için harika bir romantik kitap buldum. Hava güneşli ve senin de keyfin yerinde.  Pride and Prejudice tam sana göre!*")
col2.markdown("*Mehmet, sesli kitapları sevdiğini biliyorum. İşte işe giderken dinleyebileceğin bir kitap:   Sapiens: İnsanlığın Kısa Tarihi . Eminim çok şey öğreneceksin.*")
#col2.image("https://i.pinimg.com/564x/ca/8f/dc/ca8fdc82586bea8fa7fff647ee37b4e1.jpg")
col2.image("https://img.aydinlik.com.tr/rcman/Cw1280h720q95gc/storage/files/images/2023/07/11/amazonda-yapay-zeka-tarafindan-yazilan-kitaplar-yayginlasiyor-istila-mi-FHNP.jpg")
col3.subheader("Günün Kitap Tavsiyesi")



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


# Kullanıcıdan girdi alma
api_key = '9b0d2c746a9ee16b19a569fa9a2d05a8'  # OpenWeatherMap API anahtarınızı buraya yazın
city = col3.text_input("Şehir Adını Girin:")

# Kullanıcıdan kitap formatını seçmesini isteyin
book_format = col3.radio("Kitap formatını seçin:", ('text', 'audio'))


# Tavsiye butonu
if col3.button('Kitap Tavsiye Et', key='recommend_button'):
    if not city:
        col3.warning("Lütfen bir şehir adı girin.")
    else:
        try:
            weather_data = get_weather(api_key, city)
            
            if 'weather' in weather_data:
                weather_condition = weather_data['weather'][0]['main']
                book_recommendations = suggest_books_by_weather(weather_condition, book_format)
                
                col3.write(f"Hava Durumu: {weather_condition}")
                col3.write("Önerilen Kitaplar:")
                for book in book_recommendations:
                    col3.write(f"- {book}")
            else:
                col3.error("Hava durumu bilgisi alınamadı. API yanıtını kontrol edin.")
        except Exception as e:
            col3.error(f"Bir hata oluştu: {e}")
 

# graph tab

fig = px.bar(data_frame=top_books.head(10),
                 x='Author',
                 y="Title",
                 orientation="h",
                 hover_data=["Main Genre"],
                 color="Rating",
                 color_continuous_scale='blues')
                 
graph_tab.plotly_chart(fig)


genres = ["Arts, Film & Photography", "Children's Books", "Fantasy, Horror & Science Fiction", "Comics & Mangas", "Romance"]
selected_genre = graph_tab.selectbox(label="Tür seçiniz", options=genres)
graph_tab.markdown(f"Seçilen tür: **{selected_genre}**")

graph_tab.dataframe(
    meta.loc[
        meta['Main Genre'].str.contains(selected_genre, na=False), 
        ['Title', 'Main Genre', 'Rating']
    ].sort_values(by="Rating", ascending=False).head(10)
)


# recommendation_tab
r_col1, r_col2, r_col3 = recommendation_tab.columns([1,2,1])


def find_similar_books(book_title, meta, user_pca, top_n=5, genre=None, sub_genre=None):
    # Filtreleme işlemleri
    if genre:
        genre_meta = meta[meta['Main Genre'] == genre]
        genre_indices = genre_meta.index
        user_pca = user_pca[genre_indices]
    else:
        genre_meta = meta
    
    if sub_genre:
        genre_meta = genre_meta[genre_meta['Sub Genre'] == sub_genre]
        genre_indices = genre_meta.index
        user_pca = user_pca[genre_indices]

    # Kısmi eşleşmeyi bul ve en yakın başlığı seç
    titles = genre_meta['Title'].tolist()
    best_match = process.extractOne(book_title.strip(), titles)
    
    if best_match is None or best_match[1] < 80:  # Benzerlik skoru eşiği
        raise ValueError(f"'{book_title}' kitabı veri setinde bulunmuyor.")

    matched_title = best_match[0]
    idx = genre_meta[genre_meta['Title'] == matched_title].index[0]
    
    # Cosine benzerlik hesapla
    cosine_sim = cosine_similarity(user_pca)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    
    book_indices = [i[0] for i in sim_scores]
    similar_books = genre_meta.iloc[book_indices]
    
    # İlk kitabı önerilenlerden çıkararak tekrarını engelleme
    book_indices = [i[0] for i in sim_scores if genre_meta.iloc[i[0]]['Title'] != matched_title][:top_n]
    similar_books = genre_meta.iloc[book_indices]
    
    # Tekrar eden başlıkları kaldır
    unique_books = similar_books.drop_duplicates(subset='Title', keep='first')
    
    # Alt türlere göre filtreleme
    filtered_books = unique_books.groupby('Sub Genre').first().reset_index()
      
    return filtered_books[['Title', 'Author', 'Main Genre', 'Sub Genre', 'Price', 'URLs']]

# Streamlit uygulaması
recommendation_tab.title('Kitap Tavsiye Sistemi')

# Kullanıcıdan girdi alma
book_title = recommendation_tab.text_input("Kitap Başlığını Girin:")

# Tavsiye butonu
if recommendation_tab.button('Kitap Tavsiye Et'):
    if not book_title:
        recommendation_tab.warning("Lütfen bir kitap başlığı girin.")
    else:
        try:
            # Gerçek veri ve PCA özelliklerinin sağlandığından emin olun
            similar_books = find_similar_books(book_title, meta, user_pca)
            
            if similar_books.empty:
                recommendation_tab.write("Maalesef öneri bulunamadı.")
            else:
                recommendation_tab.write("Önerilen Kitaplar:")
                for index, row in similar_books.iterrows():
                    recommendation_tab.image(row['URLs'], caption=row['Title'])
                    recommendation_tab.write(f"Yazar: {row['Author']}")
                    recommendation_tab.write(f"Tür: {row['Main Genre']} - Alt Tür: {row['Sub Genre']}")
                    recommendation_tab.write(f"Fiyat: {row['Price']}")
                    recommendation_tab.write("---")
                #recommendation_tab.write("Önerilen Kitaplar:")
                #recommendation_tab.dataframe(similar_books[['Title','Author', 'Main Genre', 'Sub Genre']])
        except ValueError as e:
            recommendation_tab.error(e)
        except Exception as e:
            recommendation_tab.error(f"Bir hata oluştu: {e}")
