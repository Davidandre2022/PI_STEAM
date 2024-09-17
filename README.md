# <h1 align=center> **PROYECTO INDIVIDUAL Nº1**
# <h1 align=center> ** ANDRE MONTES **
### <h1 align=center> `Machine Learning Operations` (MLOps)

Este proyecto se centra en la creación de un modelo de Machine Learning destinado a abordar un desafío comercial específico de Steam: la recomendación de videojuegos para usuarios. Durante el desarrollo, se llevó a cabo un trabajo de Ingeniería de Datos para crear un Producto Mínimo Viable (MVP), que se enfoca en la consulta de datos específicos y en la recomendación de juegos similares a aquellos proporcionados por el usuario de Steam.

Se ha creado una API utilizando el framework FastAPI para facilitar el acceso a los datos de la empresa. Esta API permite realizar consultas precisas en una base de datos filtrada, garantizando la integridad y calidad de los datos mediante un exhaustivo trabajo de Extracción, Transformación y Carga (ETL). El objetivo principal es ofrecer a los usuarios la capacidad de obtener información relevante y recomendaciones de juegos de manera eficiente a través de esta interfaz.

## <h1 align=center> **`Extracción, Transformación y Carga de Datos (Descripción General)`**

En el proceso de carga de datos en la carpeta ETL, se lleva a cabo la manipulación de 3 archivos JSON en Python. Se inicia convirtiendo cada línea de los archivos en diccionarios individuales, los cuales se almacenan en una lista. Posteriormente, esta lista de diccionarios se transforma en un DataFrame de pandas.

En cuanto a la limpieza y transformación de los datos en los DataFrames resultantes:

1. **Explode:** Se emplea la función `explode()` para desglosar cada elemento de una lista en una fila independiente, replicando los valores del índice. Esto se utiliza para expandir columnas con datos anidados en el DataFrame.
2. **Json Normalization:** La función `json_normalize()` es utilizada para convertir una columna específica, que contiene datos anidados en forma de lista de diccionarios, en un nuevo DataFrame. Luego, la función `set_index()` ajusta el índice de este nuevo DataFrame para que coincida con el índice de la columna original.
3. **Concatenación:** Se utiliza la función `concat()` para unir el DataFrame original con el nuevo DataFrame creado a partir de la columna con datos anidados, concatenándolos a lo largo de las columnas.
4. **Eliminación de Columnas:** La función `drop()` se emplea para eliminar columnas específicas que tienen poca relevancia para el análisis.
5. **Eliminación de Filas Faltantes:** Se utiliza `dropna()` para eliminar filas que contienen valores faltantes en las columnas seleccionadas.
6. **Selección de Columnas:** Se seleccionan las columnas finales del DataFrame.
7. **Eliminación de Duplicados:** La función `drop_duplicates()` elimina filas duplicadas del DataFrame.
8. **Exportación:** Finalmente, se exportan los DataFrames resultantes a archivos CSV (`'user_items.csv'`, `'user_reviews.csv'` y `'games.csv'`) utilizando la función `to_csv()`, con el parámetro `index=False` para evitar incluir los índices de filas en los archivos CSV.

**Condiciones específicas para el tratamiento de datos en cada datatset:**

- **Para el dataset `australian_user_items`:**
  - En el documento `ETL_items.ipynb` de la carpeta ETL:
    - `data_it3 = data_it3[data_it3['playtime_forever'] != 0]` filtra el DataFrame para incluir únicamente las filas donde la columna `'playtime_forever'` no es igual a 0. Esto significa que se consideraron solo aquellos ítems que fueron jugados durante al menos una hora.

- **Para el dataset `australian_user_reviews`:**
  - En el documento `ETL_reviews.ipynb` de la carpeta ETL:
    - `data_re2 = data_re1['reviews'].apply(pd.Series)` utiliza la función `apply()` para convertir la columna `'reviews'`, que es una lista de diccionarios, en un DataFrame. Luego, este nuevo DataFrame se concatena con el original.
    - `data_re3['year_posted'] = data_re3['posted'].str.extract('(\d{4})')` crea una nueva columna llamada `'year_posted'` extrayendo el año de la columna `'posted'` mediante una expresión regular.
    - `data_re3['recommend'] = data_re3['recommend'].replace({'False': 0, 'True': 1}).astype(int)` reemplaza los valores booleanos `'True'` y `'False'` en la columna `'recommend'` con 1 y 0, respectivamente, cambiando el tipo de datos de la columna a entero.
    - La función `get_sentiment(text)` se utiliza para analizar el sentimiento del texto utilizando TextBlob. Devuelve 0 si la polaridad del sentimiento es menor que -0.1 (negativo), 2 si es mayor que 0.1 (positivo) y 1 en caso contrario (neutral).
    - `data_re3['sentiment_analysis'] = data_re3['review'].apply(get_sentiment)` aplica la función `get_sentiment()` a la columna `'review'` y almacena los resultados en una nueva columna `'sentiment_analysis'`. Se menciona que el análisis se realizó solo en registros válidos que fueron comentados, valorados y recomendados.

- **Para el dataset `output_steam_games`:**
  - En el documento `ETL_games.ipynb` de la carpeta ETL:
    - `data_games['release_date'] = pd.to_datetime(data_games['release_date'], errors='coerce').dt.year` convierte la columna `'release_date'` a formato de fecha y extrae el año.
    - Se realiza un reemplazo de varios valores de cadena en la columna `'price'` (`'Free to Play'`, `'Free To Play'`, `'Play For Free'`) con 0.
    - `data_games = data_games[pd.to_numeric(data_games['price'], errors='coerce').notnull()]` convierte la columna `'price'` a valores numéricos y elimina filas con valores no numéricos en esa columna.
    - `data_games1 = data_games['genres'].apply(pd.Series)` utiliza la función `apply()` para dividir la columna `'genres'` en varias columnas. El DataFrame resultante se concatena con el original, conservando solo el primer valor de la lista en la columna `'genres'`, eliminando los demás "géneros" y la columna `'genres'`.

**Relación y unión de tablas:**

El siguiente fragmento de código, contenido en el archivo `ETL_TOTAL.ipynb`, lleva a cabo las etapas de limpieza, transformación y fusión de datos utilizando varios DataFrames de pandas. Aquí se presenta una explicación detallada de cada paso:

1. **Creación de Identificadores Únicos:** Se genera un identificador único denominado `'id'` en los DataFrames `'items'` y `'reviews'` mediante la concatenación de las columnas `'user_id'` y `'item_id'`.
2. **Fusión de DataFrames con la Función `merge()`:** Se utiliza la función `merge()` para combinar los DataFrames `'reviews'` y `'games'` basándose en la columna común `'item_id'`. El resultado de esta fusión se almacena en un nuevo DataFrame llamado `'merged_df'`.
3. **Fusión Adicional de DataFrames:** Se lleva a cabo una fusión adicional mediante la línea de código `steam = items.merge(merged_df, on='id')`, uniendo el DataFrame `'items'` con `'merged_df'` utilizando el identificador único `'id'`.
4. **Selección de Columnas Relevantes:** Se realiza una selección de columnas específicas del DataFrame `'steam'`, incluyendo `'id'`, `'user_id'`, `'item_id'`, `'title'`, `'genre'`, `'developer'`, `'release_date'`, `'price'`, `'recommend'`, `'year_posted'`, `'sentiment_analysis'`, y `'playtime_forever'`.
5. **Exportación a Archivo CSV:** Se utiliza la función `to_csv()` para escribir el DataFrame `'steam'` en un archivo CSV llamado `'data_steam.csv'`. La opción `index=False` indica que no se debe incluir el índice en el archivo CSV resultante.

## <h1 align=center> **`Análisis de Datos Exploratorio`** 

En el archivo EDA_PI.ipynb, se realizó un análisis exploratorio de datos (EDA) extenso sobre las variables numéricas `release_date`, `price` y `playtime_forever` para identificar y eliminar outliers. Se aplicó el método del Rango Intercuartílico (IQR) en tres conjuntos de datos: `data1`, `data2` y `data3`. Utilizando Q1, Q3 y la fórmula `IQR = Q3 - Q1`, se calcularon los límites para cada variable y se excluyeron los valores que estaban fuera de esos rangos para evitar resultados sesgados.

Después de tratar los outliers, se observó una distribución más típica en estas variables. Además, las variables categóricas (`user_id`, `item_id`, `genre`, `developer` y `title`) se codificaron utilizando `LabelEncoder`, y los valores transformados se almacenaron en nuevas columnas (`user_id1`, `item_id1`, `genre1`, `developer1`, `title1`).

### Selección de Características:
Para evaluar la correlación de las variables con la variable objetivo (`item_id`), se emplearon dos métodos:
1. **Matriz de correlación**: Visualizada para comprender mejor las relaciones entre las variables.
2. **SelectKBest (con mutual_info_classif)**: Este método ayudó a identificar las mejores características basadas en las puntuaciones de información mutua con la variable objetivo.

Ambos enfoques destacaron que las características más significativas fueron `title`, `developer`, `release year`, `price`, `playtime` y `genre`.

## <h1 align=center> **`Machine Learning`**

### Preparación de Datos

En esta sección, se prepara la información necesaria para aplicar el método de Similitud de Coseno, enfocándose en las variables más relevantes para la recomendación de juegos. Se crean dos subconjuntos de datos:
- **juegos_id**: Contiene las columnas `item_id` y `título del juego`.
- **juegos_steam**: Contiene las columnas `item_id` y `features`, donde la columna `features` es una combinación de `title`, `developer`, y `release_date`, concatenadas en una sola columna separada por comas y espacios.

Estas variables son utilizadas para aplicar el modelo de Similitud de Coseno en el sistema de recomendación.

### Modelo de Recomendación

La función `recomendacion_juego(item_id)` recomienda juegos similares al especificado por `item_id`. Aquí está una descripción paso a paso del proceso:

```python
Lectura de Datos:
data = pd.read_csv('juegos_steam.csv')
data_juegos_steam = pd.read_csv('juegos_id.csv')

Vectorización de Texto:
tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b'): Se inicializa un TfidfVectorizer que convierte el texto en vectores de características. Define la frecuencia mínima y máxima de documentos para filtrar términos.
data_vector = tfidv.fit_transform(data['features']): Ajusta el TfidfVectorizer a la columna 'features' del DataFrame data y transforma las 'features' en una matriz de TF-IDF.
data_vector_df = pd.DataFrame(data_vector.toarray(), index=data['item_id'], columns=tfidv.get_feature_names_out()): Convierte la matriz de características TF-IDF en un DataFrame.

Cálculo de Similitud de Coseno:
vector_similitud_coseno = cosine_similarity(data_vector_df.values): Calcula la similitud del coseno entre todos los pares de vectores TF-IDF.
cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index): Convierte la matriz de similitud de coseno en un DataFrame.

Selección y Ordenamiento de Juegos Similares:
juego_simil = cos_sim_df.loc[item_id]: Selecciona la fila correspondiente al 'item_id' del juego de entrada en el DataFrame de similitud de coseno.
simil_ordenada = juego_simil.sort_values(ascending=False): Ordena la fila seleccionada en orden descendente de similitud de coseno.
resultado = simil_ordenada.head(6).reset_index(): Selecciona los 6 juegos más similares y restablece el índice del DataFrame resultante.

Obtención de Títulos de Juegos Recomendados:
result_df = resultado.merge(data_juegos_steam, on='item_id', how='left'): Fusiona el DataFrame resultante con el DataFrame 'data_juegos_steam' basado en la columna 'item_id' para obtener los títulos de los juegos recomendados.

Generación del Mensaje de Recomendación:
Las siguientes tres líneas crean un mensaje recomendando los 5 mejores juegos más similares al juego de entrada y almacenan el mensaje y los juegos recomendados en un diccionario.

Retorno de Resultados:
return result_dict: Devuelve el diccionario que contiene el mensaje y los juegos recomendados.
```

## <h1 align=center> **`Funciones y API`**

Esta aplicación FastAPI permite realizar diversas consultas y análisis sobre un conjunto de datos de juegos provenientes de la plataforma Steam. Los datos se cargan desde un archivo CSV y se procesan utilizando la biblioteca pandas. La aplicación responde a solicitudes `GET` en varios puntos finales, cada uno diseñado para realizar un tipo específico de análisis sobre los datos.

### Puntos Finales

#### 1. `PlayTimeGenre(genero: str)`

Este punto final devuelve el año con más horas jugadas para el género proporcionado.

- **Ejemplo de retorno**:
  ```json
  {
    "Año de lanzamiento con más horas jugadas para Género X": 2013
  }
  
#### 2. `UserForGenre(genero: str)`

Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.

- **Ejemplo de retorno**:
  ```json
  {
  "Usuario con más horas jugadas para Género X": "us213ndjss09sdf",
  "Horas jugadas": [
    {"Año": 2013, "Horas": 203},
    {"Año": 2012, "Horas": 100},
    {"Año": 2011, "Horas": 23}
  ]
  }

#### 3. `UsersRecommend( año : int )`

Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)

- **Ejemplo de retorno**:
  ```json
  [
  {"Puesto 1": "Juego X"},
  {"Puesto 2": "Juego Y"},
  {"Puesto 3": "Juego Z"}
  ]

#### 4. `UsersWorstDeveloper( año : int )`

Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)

- **Ejemplo de retorno**:
  ```json
  [
  {"Puesto 1": "Desarrolladora X"},
  {"Puesto 2": "Desarrolladora Y"},
  {"Puesto 3": "Desarrolladora Z"}
  ]
  
#### 4. `Sentiment_analysis( empresa desarrolladora : str )`

Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor.

- **Ejemplo de retorno**:
  ```json
  {
  "Valve": [
    {"Negative": 182},
    {"Neutral": 120},
    {"Positive": 278}
  ]
  }
Cada función de punto final lee el conjunto de datos del archivo CSV, realiza procesamiento utilizando pandas y devuelve una respuesta, generalmente en forma de diccionario o lista de diccionarios que contiene los resultados del análisis.

Es importante tener en cuenta que las consultas deben realizarse respetando las mayúsculas y minúsculas en el campo correspondiente para obtener resultados efectivos. Por ejemplo, al ingresar datos para la primera función, se debe proporcionar el nombre del desarrollador como ´´Valve´´; escribirlo como ´´valve´´ (en minúsculas) no generará una respuesta.

Enlace de deployment: https://modelo-steam17.onrender.com/docs
