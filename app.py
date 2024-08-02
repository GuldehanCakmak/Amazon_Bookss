import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def get_data():
    meta = pd.read_csv('data/streamlit_Books_df.csv')
    return meta

meta = get_data()

st.title(':blue[Miuul] Movie :blue[Recommender] 🎥', )

home_tab, graph_tab, recommendation_tab = st.tabs(["Ana Sayfa", "Grafikler","Öneri Sistemi"])

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




