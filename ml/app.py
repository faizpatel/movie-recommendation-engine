from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import pickle
import requests

app = Flask(__name__)

titles = pd.read_csv('titles.csv')
titles_list = titles['title'].tolist()
movies = pd.read_csv('tmdb_5000_movies.csv')
movie_titles = movies['title'].tolist()
plot_overviews = movies[['title','overview','homepage']]
df = pd.read_csv('ml_output.csv')
similarity = pickle.load(open('sim.pkl','rb'))

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

def recommend(title):
    reco = []
    id = []
    plots = []
    homepages = []
    movie_index = get_index_from_title(title)
    cosine_scores = similarity[movie_index]
    simmovies = list(enumerate(cosine_scores))
    sorted_movies = sorted(simmovies, key = lambda x:x[1],reverse=True)
    x = 0
    for movie in sorted_movies:
        plots.append(get_plot_from_title(get_title_from_index(movie[0])))
        reco.append(get_title_from_index(movie[0]))
        id.append(get_id_from_index(movie[0]))
        homepages.append(get_hp_from_title(get_title_from_index(movie[0])))
        if x > 4:
            break
        x = x + 1
    return reco,id,plots,homepages

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=25f03fbab6a3738b0be8b0299f4cd1fa&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

@app.route('/home', methods=["POST","GET"])
@app.route('/',methods=["POST","GET"])
def index():
    if request.method == "POST":
        user = request.form["movie_input"]
        return redirect(url_for("reco", movie_name=user))
    else:
        suggestions = titles_list
        return render_template('home.html',suggestions = suggestions)



@app.route('/<movie_name>')
def reco(movie_name):
    movie_name = movie_name.replace("`","'")
    movie_name = movie_name.replace("(and)", "&")
    movies = recommend(movie_name)[0]
    id = recommend(movie_name)[1]
    posters = []
    plots = recommend(movie_name)[2]
    homepages = recommend(movie_name)[3]
    for i in range(0,6):
        if(type(homepages[i]) != type("str")):
            homepages[i] = "https://www.imdb.com/"
    for i in id:
      posters.append(fetch_poster(i))
    return render_template('reco.html',movies = movies, posters=posters, movie_name = movie_name, plots = plots, homepages=homepages)

@app.route('/aboutsss')
def about_page():
    return render_template('about.html')

if __name__ == "__main__":
    #app.run(int(os.environ.get('PORT', 33507)))
    app.run(debug=True)