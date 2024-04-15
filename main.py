
#%pprint
import pandas as pd #importa la paquetería PAnel DAta (llamada pandas)
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted # importa paqueteria para graficar diagramas de venn
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt #importa pyplot para hacer gáficas
from matplotlib import numpy as np #importar numpy
import seaborn as sn
import altair as alt
#!pip install altair_catplot
import altair_catplot as altcat
#!pip install xlsxwriter
import xlsxwriter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

import streamlit as st
import pandas as pd
df=pd.read_excel('AM_2023_Antropo.xlsx')
st.dataframe(df)
df=df.dropna()
df['FA'] = (df['Fuerza mano derecha'] + df['Fuerza mano izquierda']) / 2
df['Gs Brazo'] = (df['Gs Brazo derecho'] + df['Gs Brazo izquierdo']) / 2
df['Gs Pierna'] = (df['Gs pierna derecha'] + df['Gs pierna izquierda']) / 2
df=df[['Folio', 'Peso (kg)', 'Talla (cm)', 'IMC', 'PCintura (cm)',
       'PCadera (cm)', 'PBrazo (cm)', 'PPantorrilla (cm)', 'PCB (mm)',
       'PCT (mm)', 'PCSE (mm)', 'Agua Corporal (%)', 'Músculo (kg)',
       'Grasa Corporal (%)', 'Centro',
       'FA','Velocidad de marcha']]
df_2=df

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Carga tus datos desde reduced_df_2 (reemplaza 'data.csv' con tu propio DataFrame)
# Considera cargar tus datos desde un archivo CSV o mediante la carga directa de datos.
# data = pd.read_csv('data.csv')

# Puedes definir tus datos de prueba aquí para simular la carga de datos.
data = pd.DataFrame({'PPantorrilla (cm)': np.random.rand(100) * 50,
                     'Músculo (kg)': np.random.rand(100) * 100})

# Divide tus datos en características (X) y la variable dependiente (y)
X = data[['PPantorrilla (cm)']]
y = data['Músculo (kg)']

# Crea un título para la aplicación
st.title('Modelos de Regresión para Predicción de Músculo')

# Muestra una tabla con los primeros registros de los datos
st.write("Primeros registros de los datos:")
st.write(data.head())

# Crea un modelo de regresión lineal
modelo_musculo_lr = LinearRegression()
modelo_musculo_lr.fit(X, y)

# Crea un modelo de árbol de decisión
modelo_musculo_dt = DecisionTreeRegressor()
modelo_musculo_dt.fit(X, y)

# Crea un modelo de Random Forest
modelo_musculo_rf = RandomForestRegressor()
modelo_musculo_rf.fit(X, y)

# Realiza predicciones para diferentes valores de PPantorrilla (cm)
ppantorilla_values = np.linspace(min(X['PPantorrilla (cm)']), max(X['PPantorrilla (cm)']), 100).reshape(-1, 1)
musculo_pred_lr = modelo_musculo_lr.predict(ppantorilla_values)
musculo_pred_dt = modelo_musculo_dt.predict(ppantorilla_values)
musculo_pred_rf = modelo_musculo_rf.predict(ppantorilla_values)

# Calcula el coeficiente de determinación (R^2) para cada modelo
r2_musculo_lr = modelo_musculo_lr.score(X, y)
r2_musculo_dt = modelo_musculo_dt.score(X, y)
r2_musculo_rf = modelo_musculo_rf.score(X, y)

# Grafica los datos y las predicciones para cada modelo
st.write("Gráfico de predicciones:")
plt.scatter(X, y, label='Datos reales')
plt.plot(ppantorilla_values, musculo_pred_lr, label=f'Regresión Lineal (R^2={r2_musculo_lr:.2f})', color='red')
plt.plot(ppantorilla_values, musculo_pred_dt, label=f'Árbol de Decisión (R^2={r2_musculo_dt:.2f})', color='green')
plt.plot(ppantorilla_values, musculo_pred_rf, label=f'Random Forest (R^2={r2_musculo_rf:.2f})', color='blue')
plt.xlabel('PPantorrilla (cm)')
plt.ylabel('Músculo (kg)')
plt.legend()
st.pyplot()

# Coeficientes de ajuste para el modelo de regresión lineal
pendiente_musculo_lr = modelo_musculo_lr.coef_[0]
intercepto_musculo_lr = modelo_musculo_lr.intercept_
st.write(f'Ajuste Lineal: Pendiente = {pendiente_musculo_lr}, Intercepto = {intercepto_musculo_lr}')

# Coeficientes de determinación (R^2) para los modelos
st.write(f'R^2 Ajuste Lineal: {r2_musculo_lr}')
st.write(f'R^2 Árbol de Decisión: {r2_musculo_dt}')
st.write(f'R^2 Random Forest: {r2_musculo_rf}')
