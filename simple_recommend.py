import pandas as pd

# load metadata
metadata = pd.read_csv('data/movies_metadata.csv', low_memory=False)

# Print the first three rows
print(metadata.head(3))

# calculate mean of vote average column - средний рейтинг по всем фильмам
c = metadata['vote_average'].mean()
print(f"average rating: {round(c, 2)}")


# calculate the minimum number of voters required to be in the chart
# из 90% списка фильмов находится минимальное количество голосовавших
# полученное значение является границей входа в список рекомендаций
m = metadata['vote_count'].quantile(0.90)
print(f"the min value: {round(m, 2)}")


# filter out all movies into a new DataFrame
# pandas.DataFrame.loc[index label] - извлечение данных из набора данных
q_movies=metadata.copy().loc[metadata['vote_count']>=m]
# .shape показывает (n_rows, n_columns)
print(f'(n_rows, n_columns) of the filtered DataFrame: {q_movies.shape}')
print(f'(n_rows, n_columns) of the initial DataFrame: {metadata.shape}')

# func that computes the weighted rating of each movie
def weighted_raiting(x, m=m, c=c):
    v=x['vote_count']
    R=x['vote_average']
    # calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * c)

# print(weighted_raiting(q_movies))


# define a new feature 'score' and calculate its value with weighted_raiting()
# axis = 1 or ‘columns’: apply function to each row.
q_movies['score']=q_movies.apply(weighted_raiting, axis=1)

# sort movies based on score calculated above
# ascending - sort asc (по возрастанию)
q_movies = q_movies.sort_values('score', ascending=False)

# print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))