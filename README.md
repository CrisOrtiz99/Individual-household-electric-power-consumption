## Proyecto y Objetivo 

El presente proyecto busca recrear y aprender los distintos modelos de redes neuronales aplicados en el capitulo 18 del libro "Time Series Forecasting in Python". Observar y analizar los distintos rendimientos de cada red neuronal aplicados a la base de datos de consumo energetico.


La base de datos ocupada se dejara a continuacion:
 https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

El link de libro ocupado se dejara a continuacion:
 https://www.manning.com/books/time-series-forecasting-in-python-book?utm_source=marcopeix&utm_medium=affiliate&utm_campaign=book_peixeiro_time_10_21_21&a_aid=marcopeix&a_bid=8db7704f

## Dependencias 
Aqui se adjuntan las dependencias necesarias y ocupadas para este programa.

- numpy 1.24.3
- pandas 2.0.1
- tensorflow 2.13.0
- matplotlib 3.7.1
- sklearn 0.0.post5

## Procedimiento

1. Prereuisitos: Se descargan e instalan las dependencias necesarios para correr optimamente el programa.

2. Adquisicion y preprocesamiento: Primero se adquieren los datos ya mencionados, se comiensa por una observacion y limpieza de estos mismos, y una preparacion para poder ocuparlos. Se genera otro data set que fue modificado, posteriormente se gneraron 3 dataset distintos a partir del dataset ya tratado, para el entrenamiento, los cuales son entrenamiento (train_df.csv), testeo (test_df.csv) y la validacion (val_df.csv).

3. Entrenamiento: Despues de generar los 3 data set, entrenamos los modelos de LSTM, CNN, CNN-LSTM, AR-LSTM, Densen, Linear, Baseline-Last y Baseline-Repeat. Se grafica el rendimiento de cada uno.



## Estructura del repositorio

El repositorio está organizado de la siguiente manera:

- Data: Esta carpeta contiene los datos utilizados en el proyecto. Se puede dividir en subcarpetas, como raw_data para los datos en bruto y processed_data para los
datos preprocesados.
- Notebooks: Aquí se encuentran los notebooks de Jupyter utilizados para realizar el análisis exploratorio de datos, preprocesamiento, modelado, etc. Puedes tener
diferentes notebooks para cada etapa del proyecto.
- Src: Esta carpeta contiene los scripts o módulos de Python que contienen funciones o clases reutilizables para el preprocesamiento de datos, modelado,
evaluación, etc.
- Models: Aquí se guardan los modelos entrenados o archivos relacionados con el modelo final del proyecto, como archivos de pesos o archivos de configuración.
- Reports: En esta carpeta se incluyen los informes, análisis o visualizaciones generados durante el proyecto. Puedes tener notebooks, documentos en PDF u otros
formatos.
- Requirements.txt: Este archivo contiene las dependencias del proyecto. Puedes incluir las bibliotecas de Python utilizadas y sus versiones correspondientes.



## Pasos seguidos y resultados obtenidos

Los pasos utilizados en este proyecto para desarrollar el agente de aprendizaje por refuerzo fueron los siguientes:

## 1. Preparación de los datos

- Se importan los paquetes necesarios: `numpy`, `pandas`, `tensorflow`, `matplotlib`, `datetime`, entre otros.
- Se establece la semilla aleatoria para TensorFlow y Numpy para asegurar la reproducibilidad de los resultados.
- Se carga el archivo de datos de consumo energético en un DataFrame de Pandas.
- Se realiza una observación inicial del DataFrame para verificar la presencia de datos nulos.
- Se elimina la columna "Sub_metering_3" debido a su falta de relevancia.
- Se convierten los valores no numéricos a numéricos en las columnas a partir de la tercera columna.
- Se realiza un ploteo de los datos de "Global_active_power" para visualizar el consumo de energía a lo largo del tiempo.

## 2. Preprocesamiento de los datos

- Se combinan las columnas de fecha y hora en una sola columna "datetime" para facilitar su manipulación.
- Se cambia el intervalo de tiempo a horas y se realiza una suma del consumo energético en cada intervalo.
- Se ordena el DataFrame resultante.
- Se guarda el DataFrame procesado en un archivo CSV.

## 3. Análisis exploratorio de datos

- Se realiza una descripción estadística del DataFrame resultante.
- Se elimina la columna "Sub_metering_1" debido a su bajo aporte estadístico.
- Se aplica una Transformada Rápida de Fourier (FFT) a la columna "Global_active_power" y se plotea el espectro de frecuencia resultante.

## 4. Preparación de los conjuntos de entrenamiento, validación y prueba

- Se divide el DataFrame en conjuntos de entrenamiento, validación y prueba.
- Se realiza una normalización de los datos utilizando un escalador MinMax.
- Se guardan los conjuntos de entrenamiento, validación y prueba en archivos CSV.

## 5. Modelado y entrenamiento

- Se definen diferentes modelos de aprendizaje automático utilizando TensorFlow y Keras: Linear, Dense, LSTM, CNN, AR-LSTM.
- Se compilan y entrenan los modelos utilizando los conjuntos de entrenamiento y validación.
- Se evalúa el rendimiento de los modelos en los conjuntos de validación y prueba.

## 6. Resultados y visualización

- Se muestra una comparación del error absoluto medio (MAE) entre los diferentes modelos en los conjuntos de validación y prueba.
- Se plotean las predicciones de los modelos en los conjuntos de validación y prueba para visualizar la calidad de las predicciones.
