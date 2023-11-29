"""
Este módulo sirve como punto de entrada principal para la aplicación FastAPI.
Define las rutas API y la lógica para manejar solicitudes.
"""
from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

app = FastAPI()

# http://127.0.0.1:8000 # Ruta raiz del puerto
# @app.get("/") # Tenemos el objeto y la ("/") es la ruta raiz. Ejecuta la funcion

steam = pd.read_csv('data_steam.csv')
steam['release_date'] = steam['release_date'].astype(int)


@app.get('/Play_Time_Genre/{genero}')
def PlayTimeGenre(genero: str):
    """Debe devolver año con mas horas jugadas para dicho género."""
       
    # Filtramos los juegos por género de manera más flexible
    df_filt1 = steam[steam['genres'].str.lower().str.contains(fr'\b{genero}\b', na=False)]

    if not df_filt1.empty:
        # Convertimos 'item_id' a tipo de dato object en ambos DataFrames
        steam['item_id'] = steam['item_id'].astype(str)
        df_filt1['item_id'] = df_filt1['item_id'].astype(str)

        df_merged = pd.merge(steam, df_filt1[['item_id', 'release_year']], left_on='item_id', right_on='item_id')
        
        # Verificamos si la longitud de df_filt1 es mayor a cero, es decir, se encontraron géneros
        result = {"Año con más horas jugadas para el género {}: {}".format(genero.capitalize(), df_merged.groupby('release_year')['playtime_hours'].sum().idxmax())}
    else:
        result = {"Año con más horas jugadas para género {}: {}".format(genero.capitalize(), "Género no encontrado en la base de datos")}

    return result



@app.get('/user_for_genre/{genero}')
def user_for_genre(genero: str):
    """Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la 
    acumulación de horas jugadas por año"""

    # Filtrar el DataFrame para el género especificado
    df_filt2 = steam[steam['genre'] == genero]

    # Agrupar por 'user_id' y sumar 'playtime_forever'
    df_group2 = df_filt2.groupby('user_id').agg(
        {'playtime_forever': 'sum'}).reset_index()

    # Ordenar por 'playtime_forever'
    df_sort2 = df_group2.sort_values('playtime_forever', ascending=False)

    # Filtrar el DataFrame original para el 'user_id' con el mayor 'playtime_forever'
    df_filt21 = steam[steam['user_id'] == df_sort2.iloc[0, 0]]

    # Agrupar por 'year_posted' y sumar 'playtime_forever'
    df_group21 = df_filt21.groupby('year_posted').agg(
        {'playtime_forever': 'sum'}).reset_index()

    df_group21 = df_group21.rename(
        columns={'year_posted': 'Año', 'playtime_forever': 'Horas'})

    # Convertir el resultado a un diccionario
    result_dicc2 = {
        'usuario con mas horas jugadas para el genero ' + genero: df_sort2.iloc[0, 0],
        'horas jugadas': df_group21.to_dict('records')
    }

    return result_dicc2


@app.get('/UsersRecommend/{anio}')
def UsersRecommend(anio: int):
    """Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado"""

    # Filtrar el DataFrame para obtener solo las filas donde 'year_posted' es el año de entrada
    df_filt3 = steam[steam['year_posted'] == anio].copy()

    # Crea una nueva columna 'reviews.recommend' que es la suma de 'recommend' y 'sentiment_analysis'
    df_filt3['reviews.recommend'] = df_filt3['recommend'] + \
        df_filt3['sentiment_analysis']

    # Agrupar por 'desarrollador' y sumar 'reviews.recommend'
    df_group3 = df_filt3.groupby('developer')[
        'reviews.recommend'].sum().reset_index()

    # Ordenar por 'reviews.recommend'
    df_sort3 = df_group3.sort_values('reviews.recommend', ascending=False)

    # Crea una lista de diccionarios para los 3 principales desarrolladores
    result_list3 = [{'Puesto ' + str(i+1): df_sort3.iloc[i, 0]}
                    for i in range(3)]

    return result_list3




@app.get('/UsersWorstDeveloper/{anio}')
def UsersWorstDeveloper(anio: int):
    """Obtiene las 3 desarrolladoras con menos recomendaciones por usuarios para el año dado"""

    # Filtrar desarrolladoras para el año dado
    desarrolladoras = steam[steam['posted_year'] == int(anio)]

    if not desarrolladoras.empty:
            # Encuentra las 3 desarrolladoras con menos recomendaciones
            desarrolladoras_top3 = desarrolladoras.nsmallest(3, 'recommend')
            resultado = [{"Puesto 1": desarrolladoras_top3.iloc[0]['developer']},
                         {"Puesto 2": desarrolladoras_top3.iloc[1]['developer']},
                         {"Puesto 3": desarrolladoras_top3.iloc[2]['developer']}]
            return resultado





@app.get('/sentimentanalysis/{desarrolladora}')
def sentiment_analysis(desarrolladora: str):
    """ Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros 
    de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor."""

    # Filtrar el DataFrame para obtener solo las filas donde 'developer' es el desarrollador de entrada
    # y 'sentiment_analysis' es 2 o 0
    df_filt5 = steam[(steam['developer'] == desarrolladora) &
                     steam['sentiment_analysis'].isin([0, 2])]

    # Agrupar por 'developer' y 'sentiment_analysis', y contar 'developer'
    df_group5 = df_filt5.groupby(
        ['developer', 'sentiment_analysis']).size().reset_index(name='count')

    # Crear un diccionario con los resultados
    result_dicc5 = {
        desarrolladora: [
            'Negativo = ' +
            str(df_group5[df_group5['sentiment_analysis'] == 0]
                ['count'].values[0]),
            'Neutral = ' +
            str(df_group5[df_group5['sentiment_analysis'] == 1]
                ['count'].values[0]),
            'Positivo = ' +
            str(df_group5[df_group5['sentiment_analysis'] == 2]
                ['count'].values[0])
            
        ]
    }

    return result_dicc5




@app.get('/recomendacion_juego/{item_id}')
def recomendacion_juego(item_id : int):

    """Ingresando el id de un juego, deberíamos recibir una lista con 5 juegos recomendados para dicho juego"""

    data = pd.read_csv('juegos_steam.csv')
    data_juegos_steam = pd.read_csv('juegos_id.csv')

    tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')
    data_vector = tfidv.fit_transform(data['features'])

    data_vector_df = pd.DataFrame(data_vector.toarray(), index=data['item_id'], columns = tfidv.get_feature_names_out())

    vector_similitud_coseno = cosine_similarity(data_vector_df.values)

    cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)

    juego_simil = cos_sim_df.loc[item_id]

    simil_ordenada = juego_simil.sort_values(ascending=False)
    resultado = simil_ordenada.head(6).reset_index()

    result_df = resultado.merge(data_juegos_steam, on='item_id',how='left')

    # Obtén el título del juego de entrada.
    juego_title = data_juegos_steam[data_juegos_steam['item_id'] == item_id]['title'].values[0]

    # Crea un mensaje de salida
    message = f"Si te gustó el juego {item_id} : {juego_title}, también te pueden gustar:"

    # Crea un diccionario de retorno de la funcion
    result_dict = {
        'mensaje': message,
        'juegos_recomendados': result_df['title'][1:6].tolist()
    }

    return result_dict
