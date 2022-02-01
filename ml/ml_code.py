import pandas as pd
import ast
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv('tmdb_5000_movies - Copy.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movie_titles = movies['title'].tolist()
plot_overviews = movies[['title','overview','homepage']]
movies = movies.merge(credits, on = 'title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

def convert(obj):
    new_list =[]
    for i in ast.literal_eval(obj):
        new_list.append(i['name'])
    return new_list

def convert_cast(obj):
    new_list =[]
    x = 0
    for i in ast.literal_eval(obj):
        if x > 2:
            break
        new_list.append(i['name'])
        x = x + 1
    return new_list

def convert_crew(obj):
    new_list =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            new_list.append(i['name'])
    return new_list

movies.dropna(inplace = True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(convert_crew)

movies['overview'] = movies['overview'].str.lower()
movies['overview'] = movies['overview'].apply(lambda x: re.sub('[^a-zA-Z]', ' ',x))
movies['overview'] = movies['overview'].apply(lambda x: re.sub('\s+', ' ',x))
movies['overview'] = movies['overview'].apply(lambda x: nltk.word_tokenize(x))

stop_words = nltk.corpus.stopwords.words('english')
stop_words.append('and')
new_overview = []

for sentence in movies['overview']:
    new_list = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            new_list.append(word)
    new_overview.append(new_list)

movies['overview'] = new_overview

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

df = movies[['movie_id','title','tags']]


df['tags'] = df['tags'].apply(lambda x:" ".join(x))
df['tags'] = df['tags'].str.lower()

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    index = df.index
    x = df["title"]==title
    a = index[x]
    ai = a.tolist()
    return ai[0]

def get_id_from_index(index):
    return df[df.index == index]["movie_id"].values[0]

def get_plot_from_title(title):
    return plot_overviews[plot_overviews.title == title]['overview'].values[0]

def get_hp_from_title(title):
    return plot_overviews[plot_overviews.title == title]['homepage'].values[0]


df.to_csv('ml_output.csv')

cv = CountVectorizer(max_features = 1000, stop_words= 'english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

pickle.dump(similarity,open('sim.pkl','wb'))