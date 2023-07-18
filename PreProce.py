#paquetes a ocupar
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell
import warnings
warnings.filterwarnings('ignore')

#Parametros para Matplotlib
plt.rcParams['figure.figsize'] = (10, 7.5)
plt.rcParams['axes.grid'] = False

#fijar la semilla de TensorFlow y Numpy(importante para replicar los resultados)
tf.random.set_seed(42)
np.random.seed(42)

#Datos de Consumo energeticos, en minutos
df = pd.read_csv(r'C:\Users\crist\OneDrive\Escritorio\archivos_MDD\Proyecto_Final\household_power_consumption.txt', sep=';')
df

#OBSERVACION DE DATAFRAME
#contabilizar Datos Nulos
df.isnull().sum()

#Optenemos la distnacia entre cada valor nulo en la columna 'Sub_metering_3'
na_groups = df['Sub_metering_3'].notna().cumsum()[df['Sub_metering_3'].isna()]
len_consecutive_na = na_groups.groupby(na_groups).agg(len)

longest_na_gap = len_consecutive_na.max()
longest_na_gap

df = df.drop(['Sub_metering_3'], axis=1)
df.head()

df.dtypes

#convertimos los valores no numericos a numericos despues de la columna 2
cols_to_convert = df.columns[2:]

df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

#Observamos que ya se trasformaron todos los datos correctamente
df.dtypes

#Ploteo de los datos Global active power
fig, ax = plt.subplots(figsize=(13,6))

ax.plot(df['Global_active_power'])
ax.set_xlabel('Timesteps (min)')
ax.set_ylabel('Global active power (kW)')
ax.set_xlim(0, 2880)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig(r'C:\Users\crist\OneDrive\Escritorio\archivos_MDD\Proyecto_Final\Imagen1.png', dpi=300)

#Combinamos las columnas Fecha y Hora
df.loc[:,'datetime'] = pd.to_datetime(df.Date.astype(str) + ' ' + df.Time.astype(str))
df = df.drop(['Date', 'Time'], axis=1)
df.head()

#cambiamos el intervalo de tiempo a Horas sumando el consumo energetico 
hourly_df = df.resample('H', on='datetime').sum()
hourly_df.head()

#ordenamos el Dataframe
hourly_df = hourly_df.drop(hourly_df.tail(1).index)
hourly_df = hourly_df.drop(hourly_df.head(1).index)
hourly_df.head()

#info del mismo
hourly_df.shape

#Reseteamos
hourly_df = hourly_df.reset_index()
hourly_df.head()

#Grafica de Energuía segun el día con el dataframe reseteado
fig, ax = plt.subplots(figsize=(13,6))

ax.plot(hourly_df['Global_active_power'])
ax.set_xlabel('Day')
ax.set_ylabel('Global active power (kW)')
ax.set_xlim(0, 336)

plt.xticks(np.arange(0, 360, 24), ['2006-12-17', '2006-12-18', '2006-12-19', '2006-12-20', '2006-12-21', '2006-12-22', '2006-12-23', '2006-12-24', '2006-12-25', '2006-12-26', '2006-12-27', '2006-12-28', '2006-12-29', '2006-12-30', '2006-12-31'])

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig(r'C:\Users\crist\OneDrive\Escritorio\archivos_MDD\Proyecto_Final\Imagen2.png', dpi=300)



#Guardamos el dataframe ya procesado
hourly_df.to_csv(r'C:\Users\crist\OneDrive\Escritorio\archivos_MDD\Proyecto_Final\household_power_consumption-proce.txt', header=True, index=False)
hourly_df