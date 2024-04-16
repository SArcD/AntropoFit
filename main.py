
#%pprint
import pandas as pd #importa la paquetería PAnel DAta (llamada pandas)
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted # importa paqueteria para graficar diagramas de venn
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt #importa pyplot para hacer gáficas
from matplotlib import numpy as np #importar numpy
import seaborn as sn
import altair as alt
import altair_catplot as altcat
import xlsxwriter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import streamlit as st

import streamlit as st

# Configurar el esquema de colores personalizado
st.set_page_config(
    page_title="Mi Aplicación Personalizada",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # Puedes elegir entre 'wide' o 'centered'
    initial_sidebar_state="expanded",  # Puedes elegir entre 'expanded' o 'collapsed'
)

# Definir colores personalizados
background_color = "#f0f0f0"  # Gris claro para el fondo
text_color = "#000000"  # Negro para el texto
interactive_elements_color = "#007bff"  # Azul para elementos interactivos

# Configurar el estilo de la aplicación
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {background_color};
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)



import streamlit as st
import base64

# Función para obtener el valor de la cookie del navegador
def get_cookie_value(key):
    if key in st.session_state:
        try:
            value = base64.b64decode(st.session_state[key]).decode()
            return int(value)
        except:
            pass
    return 0

# Función para guardar el valor de la cookie del navegador
def set_cookie_value(key, value):
    st.session_state[key] = base64.b64encode(str(value).encode()).decode()

# Obtener el contador actual de visitas
counter = get_cookie_value("visit_counter")

# Incrementar el contador y guardar los cambios en la cookie
counter += 1
set_cookie_value("visit_counter", counter)

# Mostrar el contador en la aplicación de Streamlit
st.write(f"Esta página ha sido visitada {counter} veces.")



# Crear una barra lateral para las pestañas
pestañas = st.sidebar.radio("Selecciona una pestaña:", ("Presentación", "Modelos con una variable", "Modelos con 2 variables", "Predicción de Sarcopenia", "Equipo de trabajo"))
if pestañas == "Presentación":
       st.title("Sobre la calculadora")
       st.markdown("""
       Esta aplicación tiene como objetivo servir a personal médico para el registro, análisis de datos y cribado primario de pacientes geríatricos con riesgo de sufrir de sarcopenia
       
       """
               )

       #st.title("Acerca de Sarc-open-IA")

       st.subheader("Objetivo")
       st.markdown("""
       El objetivo de esta aplicación es asistir al personal médico en la captura, almacenamiento y análisis de datos antropométricos de adultos mayores para la determinación de dependencia funcional y sarcopenia. Es el resultado de una estancia de investigación posdoctoral, resultado de la colaboración entre el **Consejo Nacional de Humanidades Ciencia y Tecnología (CONAHCYT) y la Universidad de Colima (UCOL)** y desarrollada en **noviembre de 2023**, en la que se utilizó una base de datos antropométricos de adultos mayores para crear modelos predictivos de dependencia funcional y sarcopenia. Estos modelos representan la primera fase de una estrategia diseñada para facilitar la identificación temprana de síntomas de condiciones debilitantes en adultos mayores, utilizando técnicas de inteligencia artificial y aprendizaje automático.
       """)

       st.subheader("Ventajas y características")

       st.markdown("""

       - **Objetivo de Facilitar su Uso:** Queríamos que nuestra herramienta fuera fácil de usar para el personal médico, incluso si no estaban familiarizados con la inteligencia artificial o la programación. Para lograrlo, elegimos el lenguaje de programación Python y las plataformas Streamlit y GitHub. Estas opciones permiten una fácil visualización y manipulación de la aplicación, además de almacenar los algoritmos en la nube.

       - **Interfaz Amigable:** El resultado es una interfaz gráfica que permite a los médicos ingresar los datos antropométricos de los pacientes y ver gráficas útiles para el análisis estadístico. También ofrece un diagnóstico en tiempo real de la sarcopenia, y todo esto se hace utilizando cajas de texto y deslizadores para ingresar y manipular los datos.

       - **Accesibilidad Total:** El personal médico puede descargar de forma segura las gráficas y los archivos generados por la aplicación. Además, pueden acceder a ella desde cualquier dispositivo con conexión a internet, ya sea un teléfono celular, una computadora, tablet o laptop.
       """)

       st.subheader("Método")
       st.markdown("""
       A partir de datos registrados entre septiembre y octubre el año 2023 en una muestra de adultos mayores que residen en la Zona Metropolitana, Colima, Villa de Álvarez, México, se procedió al desarrollo de modelos predictivos mediante el algoritmo Random Forest. En este caso, se crearon modelos que permiten estimar la masa muscular (medida en kilogramos) y el porcentaje corporal de grasa a partir de distintas medidas antropométricas. Estos modelos muestran un grado aceptable de coincidencia con las mediciones de estos parámetros, que típicamente requieren de balanzas de bioimpedancia y/o absorciometría de rayos X de energía dual. Una vez con las aproximaciones para masa muscular y porcentaje de grasa corporal, se estima el grado de riesgo de padecer sarcopenia para cada paciente mediante el uso del algoritmo de clustering jerarquico. Estas condiciones de diagnóstico fueron propuestas con el objetivo de minimizar la cantidad de parámetros antropométricos y establecer puntos de corte que puedan ser validados por personal médico capacitado. Este enfoque se asemeja a lo que se conoce en inteligencia artificial como un sistema experto, ya que los modelos resultantes requieren validación por parte de especialistas.
       """
       )




# Contenido de la pestaña 1
if pestañas == "Modelos con una variable":
       st.title("Modelos de aproximación de una sola variable independiente")
       st.markdown("""
       En esta pestaña se muestra el proceso para calcular modelos de aproximación a la masa muscular, medida en kilogramos, y el porcentaje de grasa corporal a partir de una sola variable. En el caso de la masa muscular, se predicen valores para pacientes  a partir del perímetro de pantorrilla y en el caso de la grasa corporal se utiliza el perimetro de brazo
       """)
       st.markdown("""
       A continuación se muestra la base de datos de adultos mayores. En la parte superior de cada columna se muestra el nombre del parámetro y las unidades correspondientes. Si deja el ícono del mouse en la parte superior derecha puede descargar la tabla con los datos.
       """)

       #import streamlit as st       
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
       st.set_option('deprecation.showPyplotGlobalUse', False)

       ########################
       data = df
       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)']]
       y = data['Músculo (kg)']

       # Crea un título para la aplicación
       st.title('Modelos de Regresión para Predicción de Músculo')

       # Muestra una tabla con los primeros registros de los datos
       st.markdown("""
       Esta es la base de datos con parámetros antropométricos:
       """)       #st.write(data.head())
       st.dataframe(data)


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
       ppantorrilla_values = np.linspace(min(X['PPantorrilla (cm)']), max(X['PPantorrilla (cm)']), 100).reshape(-1, 1)
       musculo_pred_lr = modelo_musculo_lr.predict(ppantorrilla_values)
       musculo_pred_dt = modelo_musculo_dt.predict(ppantorrilla_values)
       musculo_pred_rf = modelo_musculo_rf.predict(ppantorrilla_values)

       # Calcula el coeficiente de determinación (R^2) para cada modelo
       r2_musculo_lr = modelo_musculo_lr.score(X, y)
       r2_musculo_dt = modelo_musculo_dt.score(X, y)
       r2_musculo_rf = modelo_musculo_rf.score(X, y)

       # Grafica los datos y las predicciones para cada modelo
       st.write("Gráfico de predicciones:")
       st.write("En esta gráfica se comparan los modelos con los datos medidos (puntos azule). Las curvas de distintos colores correponden a: modelo lineal (en rojo), aproximación de Random Forest (azul) y aproximación de árbol de decisión (verde)")
       fig, ax=plt.subplots()
       ax.scatter(X, y, color = 'blue', label="Datos de masa muscular (kg)")       
       ax.plot(ppantorrilla_values, musculo_pred_lr, color='red', label=f'Regresión lineal (R^2={r2_musculo_lr:.2f})')
       ax.plot(ppantorrilla_values, musculo_pred_dt, color='green', label=f'Árbol de decisión (R^2={r2_musculo_dt:.2f})')
       ax.plot(ppantorrilla_values, musculo_pred_rf, color='blue', label=f'Random forest (R^2={r2_musculo_rf:.2f})')
       # Modificar el tamaño de fuente de las etiquetas de las líneas en el gráfico
       for label in ax.get_xticklabels() + ax.get_yticklabels():
           label.set_fontsize(8)

       ax.set_xlabel('Pantorrilla (cm)')
       ax.set_ylabel('Masa muscular (Kg)')
       ax.set_title('Predicciones de Masa muscular (Kg)')
       ax.legend(fontsize='medium')  # Modifica el tamaño de letra de las leyendas
       st.pyplot(fig)

       # Coeficientes de ajuste para el modelo de regresión lineal       
       pendiente_musculo_lr = modelo_musculo_lr.coef_[0]
       intercepto_musculo_lr = modelo_musculo_lr.intercept_
       st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_musculo_lr},** Intercepto** = {intercepto_musculo_lr}')

       # Coeficientes de determinación (R^2) para los modelos
       st.write(f'**R^2 Ajuste Lineal:** {r2_musculo_lr}')       
       st.write(f'**R^2 Árbol de Decisión:** {r2_musculo_dt}')
       st.write(f'**R^2 Random Forest:** {r2_musculo_rf}')

       import streamlit as st
       import matplotlib.pyplot as plt
       from sklearn.tree import DecisionTreeRegressor, plot_tree

       # Crear un modelo de árbol de decisión limitando la profundidad
       modelo_musculo_dt_simplified = DecisionTreeRegressor(max_depth=4)  # Ajusta el valor de max_depth según sea necesario
       modelo_musculo_dt_simplified.fit(X, y)

       # Generar el diagrama del árbol de decisión simplificado
       fig = plt.figure(figsize=(20, 10))
       plt.rc('font', size=12)  # Ajusta el tamaño de fuente aquí
       plot_tree(modelo_musculo_dt_simplified, filled=True, feature_names=X.columns)
       plt.title("Árbol de Decisión Simplificado para Musculo (kg) vs. PPantorrilla (cm)", fontsize=24)  # Ajusta el tamaño de fuente del título aquí

       # Mostrar la figura en Streamlit
       st.pyplot(fig)

       ####################


       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor
       from sklearn.metrics import r2_score

       data = df
       # Crear un modelo de regresión lineal para Grasa Corporal (%) vs. PBrazo (cm)
       X_grasa = data[['PBrazo (cm)']]
       y_grasa = data['Grasa Corporal (%)']
       modelo_grasa_lr = LinearRegression()
       modelo_grasa_lr.fit(X_grasa, y_grasa)
       # Crear un modelo de árbol de decisión para Grasa Corporal (%) vs. PBrazo (cm)
       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X_grasa, y_grasa)
       # Crear un modelo de Random Forest para Grasa Corporal (%) vs. PBrazo (cm)
       modelo_grasa_rf = RandomForestRegressor()
       modelo_grasa_rf.fit(X_grasa, y_grasa)

       # Predicciones para Grasa Corporal (%)
       pbrazo_value = 30  # Cambia este valor por el que desees predecir

       grasa_pred_lr = modelo_grasa_lr.predict(np.array([[pbrazo_value]]))
       grasa_pred_dt = modelo_grasa_dt.predict(np.array([[pbrazo_value]]))
       grasa_pred_rf = modelo_grasa_rf.predict(np.array([[pbrazo_value]]))

       # Coeficientes de ajuste para el modelo de regresión lineal
       pendiente_grasa_lr = modelo_grasa_lr.coef_[0]
       intercepto_grasa_lr = modelo_grasa_lr.intercept_

       # Coeficientes de determinación (R^2) para el modelo de regresión lineal
       r2_grasa_lr = r2_score(y_grasa, modelo_grasa_lr.predict(X_grasa))

       # Coeficientes de determinación (R^2) para el modelo de árbol de decisión
       r2_grasa_dt = modelo_grasa_dt.score(X_grasa, y_grasa)

       # Coeficientes de determinación (R^2) para el modelo de Random Forest
       r2_grasa_rf = modelo_grasa_rf.score(X_grasa, y_grasa)

       # Predicciones para Grasa Corporal (%) usando árbol de decisión y Random Forest
       X_pred_grasa_dt = np.linspace(min(X_grasa['PBrazo (cm)']), max(X_grasa['PBrazo (cm)']), 100).reshape(-1, 1)
       y_pred_grasa_dt = modelo_grasa_dt.predict(X_pred_grasa_dt)
       y_pred_grasa_rf = modelo_grasa_rf.predict(X_pred_grasa_dt)

       # Visualización de las predicciones
       st.title('Predicciones de Grasa Corporal (%)')
       st.write("Gráfico de predicciones:")
       fig, ax = plt.subplots()
       ax.scatter(X_grasa, y_grasa, color='blue', label='Datos de Grasa Corporal (%)')
       ax.plot(X_pred_grasa_dt, y_pred_grasa_dt, color='red', label=f'Árbol de Decisión (R^2={r2_grasa_dt:.2f})')
       ax.plot(X_pred_grasa_dt, y_pred_grasa_rf, color='green', label=f'Random Forest (R^2={r2_grasa_rf:.2f})')
       ax.plot(X_grasa, modelo_grasa_lr.predict(X_grasa), color='orange', label=f'Regresión Lineal (R^2={r2_grasa_lr:.2f})')
       for label in ax.get_xticklabels() + ax.get_yticklabels():
           label.set_fontsize(8)
       ax.set_xlabel('PBrazo (cm)')
       ax.set_ylabel('Grasa Corporal (%)')
       ax.set_title('Predicciones de Grasa Corporal (%)')
       ax.legend()
       st.pyplot(fig)

       # Coeficientes de ajuste para el modelo de regresión lineal
       pendiente_grasa_lr = modelo_grasa_lr.coef_[0]
       intercepto_grasa_lr = modelo_grasa_lr.intercept_
       st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_grasa_lr}, **Intercepto** = {intercepto_grasa_lr}')

       # Coeficientes de determinación (R^2) para los modelos
       st.write(f'**R^2 Ajuste Lineal:** {r2_grasa_lr}')
       st.write(f'**R^2 Árbol de Decisión:** {r2_grasa_dt}')
       st.write(f'**R^2 Random Forest:** {r2_grasa_rf}')

       #######################

       import streamlit as st
       from sklearn.tree import DecisionTreeRegressor
       import matplotlib.pyplot as plt
       from sklearn.tree import plot_tree

       # Crear un modelo de árbol de decisión limitando la profundidad
       modelo_grasa_dt_simplified = DecisionTreeRegressor(max_depth=4)  # Ajusta el valor de max_depth según sea necesario
       modelo_grasa_dt_simplified.fit(X_grasa, y_grasa)

       # Generar el diagrama del árbol de decisión simplificado
       #st.set_option('deprecation.showPyplotGlobalUse', False)  # Deshabilitar el warning sobre el uso de plt.pyplot
       plt.figure(figsize=(20, 10))
       plt.rc('font', size=12)  # Ajusta el tamaño de fuente aquí
       plot_tree(modelo_grasa_dt_simplified, filled=True, feature_names=X_grasa.columns)
       ax.set_title("Árbol de Decisión Simplificado para Grasa Corporal (%) vs. PBrazo (cm)", fontsize=24)
       st.pyplot()

       ##################################
# Contenido de la pestaña 2
elif pestañas == "Modelos con 2 variables":
       st.subheader("Contenido de la Pestaña 2")
       st.write("Aquí puedes agregar contenido para la segunda pestaña.")
       #import streamlit as st       
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
       data = df
       import streamlit as st
       import pandas as pd
       from sklearn.linear_model import LinearRegression
       from sklearn.metrics import r2_score
       import numpy as np
       import matplotlib.pyplot as plt
       from mpl_toolkits.mplot3d import Axes3D

       # Divide tus datos en características (X) y la variable dependiente (y)       
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Crea un modelo de regresión lineal
       modelo_musculo_lr = LinearRegression()
       modelo_musculo_lr.fit(X, y)

       # Realiza predicciones para diferentes valores de PPantorrilla (cm) y FA
       ppantorilla_values = np.linspace(min(X['PPantorrilla (cm)']), max(X['PPantorrilla (cm)']), 100)
       fa_values = np.linspace(min(X['FA']), max(X['FA']), 100)
       ppantorilla_values, fa_values = np.meshgrid(ppantorilla_values, fa_values)
       musculo_pred_lr = modelo_musculo_lr.predict(np.column_stack((ppantorilla_values.ravel(), fa_values.ravel())))

       # Calcula el coeficiente de determinación (R^2) para el modelo
       r2_musculo_lr = modelo_musculo_lr.score(X, y)

       # Crear una figura 3D
       fig = plt.figure(figsize=(12, 6))
       ax = fig.add_subplot(111, projection='3d')

       # Grafica los datos reales
       scatter = ax.scatter(X['PPantorrilla (cm)'], X['FA'], y, label='Datos reales', c='blue')

       # Grafica las predicciones del modelo
       ax.plot_surface(ppantorilla_values, fa_values, musculo_pred_lr.reshape(ppantorilla_values.shape), alpha=0.5, color='green')

       # Etiquetas de los ejes
       ax.set_xlabel('PPantorrilla (cm)')
       ax.set_ylabel('FA')
       ax.set_zlabel('Músculo (kg)')

       # Crear una leyenda ficticia para el gráfico
       legend = ax.legend(*scatter.legend_elements(), title="Datos reales")
       ax.add_artist(legend)

       # Título del gráfico
       plt.title(f'Músculo en función de PPantorrilla y FA (R^2={r2_musculo_lr:.2f})')

       # Mostrar el gráfico en Streamlit
       st.pyplot(fig)

       ################################

       import streamlit as st       
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor
       import numpy as np
       import matplotlib.pyplot as plt
       from mpl_toolkits.mplot3d import Axes3D

       # Carga tus datos desde reduced_df_2 (reemplaza 'data.csv' con tu propio DataFrame)
       # Asegúrate de tener definida la variable 'data'

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Crea un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor()
       modelo_musculo_dt.fit(X, y)

       # Realiza predicciones para diferentes valores de PPantorrilla (cm) y FA
       ppantorilla_values = np.linspace(min(X['PPantorrilla (cm)']), max(X['PPantorrilla (cm)']), 100)
       fa_values = np.linspace(min(X['FA']), max(X['FA']), 100)
       ppantorilla_values, fa_values = np.meshgrid(ppantorilla_values, fa_values)
       musculo_pred_dt = modelo_musculo_dt.predict(np.column_stack((ppantorilla_values.ravel(), fa_values.ravel())))

       # Calcula el coeficiente de determinación (R^2) para el modelo
       r2_musculo_dt = modelo_musculo_dt.score(X, y)

       # Crear una figura 3D
       fig = plt.figure(figsize=(12, 6))
       ax = fig.add_subplot(111, projection='3d')

       # Grafica los datos reales
       scatter = ax.scatter(X['PPantorrilla (cm)'], X['FA'], y, label='Datos reales', c='blue')

       # Grafica las predicciones del modelo de árbol de decisión
       ax.plot_surface(ppantorilla_values, fa_values, musculo_pred_dt.reshape(ppantorilla_values.shape), alpha=0.5, color='green')

       # Etiquetas de los ejes
       ax.set_xlabel('PPantorrilla (cm)')
       ax.set_ylabel('FA')
       ax.set_zlabel('Músculo (kg)')

       # Crear una leyenda ficticia para el gráfico
       legend = ax.legend(*scatter.legend_elements(), title="Datos reales")
       ax.add_artist(legend)

       # Título del gráfico
       plt.title(f'Músculo en función de PPantorrilla y FA (R^2={r2_musculo_dt:.2f})')

       # Mostrar el gráfico en Streamlit
       st.pyplot(fig)

       #############################

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.tree import DecisionTreeRegressor

       # Carga tus datos y define tus características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Entrena un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor()
       modelo_musculo_dt.fit(X, y)

       # Define rangos de valores para PPantorrilla (cm) y FA
       ppantorilla_range = np.linspace(X['PPantorrilla (cm)'].min(), X['PPantorrilla (cm)'].max(), 100)
       fa_range = np.linspace(X['FA'].min(), X['FA'].max(), 100)
       ppantorilla_grid, fa_grid = np.meshgrid(ppantorilla_range, fa_range)

       # Combina las características en una matriz bidimensional
       X_grid = np.c_[ppantorilla_grid.ravel(), fa_grid.ravel()]

       # Realiza predicciones en la malla de valores
       y_pred = modelo_musculo_dt.predict(X_grid)
       y_pred = y_pred.reshape(ppantorilla_grid.shape)

       # Crea la gráfica de superficie de decisión
       fig = plt.figure(figsize=(10, 6))
       contour = plt.contourf(ppantorilla_grid, fa_grid, y_pred, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
       plt.scatter(X['PPantorrilla (cm)'], X['FA'], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
       plt.xlabel('PPantorrilla (cm)')
       plt.ylabel('FA')
       plt.title(f'Superficie de Decisión del Árbol de Decisión para Músculo (kg) (R^2={modelo_musculo_dt.score(X, y):.2f})')

       # Agrega etiquetas con los valores de Músculo
       cbar = plt.colorbar(contour)
       cbar.set_label('Músculo (kg)')

       # Mostrar el gráfico en Streamlit
       st.pyplot(fig)


       ##################

       import streamlit as st
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor
       import matplotlib.pyplot as plt

       # Carga tus datos desde reduced_df_2 (reemplaza 'data.csv' con tu propio DataFrame)

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Crea un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor(max_depth=4)
       modelo_musculo_dt.fit(X, y)

       # Genera el diagrama del árbol de decisión
       fig = plt.figure(figsize=(50, 20))
       from sklearn.tree import plot_tree
       plot_tree(modelo_musculo_dt, filled=True, feature_names=X.columns, fontsize=20)
       plt.title("Árbol de Decisión para Músculo (kg) vs. PPantorrilla (cm) y FA", fontsize=24)

       # Mostrar el diagrama del árbol en Streamlit
       st.pyplot(fig)

       ##################

       import streamlit as st
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.tree import export_text

# Carga tus datos desde reduced_df_2 (reemplaza 'data.csv' con tu propio DataFrame)

# Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Crea un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor(max_depth=4)
       modelo_musculo_dt.fit(X, y)

       # Extraer las reglas de decisión
       tree_rules = export_text(modelo_musculo_dt, feature_names=list(X.columns))

       # Muestra las reglas de decisión en Streamlit
       st.text("Reglas de Decisión:")
       st.text(tree_rules)

       # Programa para que el usuario ingrese valores y obtenga una predicción
       def predecir_musculo(ppantorrilla, fa):
           # Realiza una predicción utilizando el modelo de árbol de decisión
           predicción = modelo_musculo_dt.predict([[ppantorrilla, fa]])[0]
           return predicción

       # Interfaz de usuario para que el usuario ingrese valores y obtenga predicciones
       ppantorrilla_input = st.number_input("Ingresa el valor de PPantorrilla (cm): ")
       fa_input = st.number_input("Ingresa el valor de FA: ")

       if st.button("Predecir Músculo", key="predict_button"):
              predicción = predecir_musculo(ppantorrilla_input, fa_input)
              st.write(f"Predicción de Músculo (kg): {predicción:.2f}")


       ###################
       import streamlit as st
       import pandas as pd
       from sklearn.linear_model import LinearRegression
       import numpy as np
       import matplotlib.pyplot as plt
       from mpl_toolkits.mplot3d import Axes3D

       X = data[['PBrazo (cm)', 'PCB (mm)']]
       y = data['Grasa Corporal (%)']

       # Crea un modelo de regresión lineal
       modelo_grasa_lr = LinearRegression()
       modelo_grasa_lr.fit(X, y)

       # Realiza predicciones para diferentes valores de PBrazo (cm) y PCB (mm)
       pbrazo_values = np.linspace(min(X['PBrazo (cm)']), max(X['PBrazo (cm)']), 100)
       pcb_values = np.linspace(min(X['PCB (mm)']), max(X['PCB (mm)']), 100)
       pbrazo_values, pcb_values = np.meshgrid(pbrazo_values, pcb_values)
       grasa_pred_lr = modelo_grasa_lr.predict(np.column_stack((pbrazo_values.ravel(), pcb_values.ravel())))

       # Calcula el coeficiente de determinación (R^2) para el modelo
       r2_grasa_lr = modelo_grasa_lr.score(X, y)

       # Crear una figura 3D para visualizar los datos y las predicciones del modelo
       fig = plt.figure(figsize=(12, 6))
       ax = fig.add_subplot(111, projection='3d')

       # Grafica los datos reales
       scatter = ax.scatter(X['PBrazo (cm)'], X['PCB (mm)'], y, label='Datos reales', c='blue')

       # Grafica las predicciones del modelo
       ax.plot_surface(pbrazo_values, pcb_values, grasa_pred_lr.reshape(pbrazo_values.shape), alpha=0.5, color='green')

       # Etiquetas de los ejes
       ax.set_xlabel('PBrazo (cm)')
       ax.set_ylabel('PCB (mm)')
       ax.set_zlabel('Grasa Corporal (%)')

       # Crear una leyenda ficticia para el gráfico
       legend = ax.legend(*scatter.legend_elements(), title="Datos reales")
       ax.add_artist(legend)

       # Título del gráfico
       plt.title(f'Grasa corporal en función de PBrazo y PCB (R^2={r2_grasa_lr:.2f})')

       # Mostrar el gráfico en Streamlit
       st.pyplot(fig)
       
       
       ###################

       import streamlit as st
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor
       import numpy as np
       import matplotlib.pyplot as plt
       from mpl_toolkits.mplot3d import Axes3D

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PBrazo (cm)', 'PCB (mm)']]
       y = data['Grasa Corporal (%)']

       # Crea un modelo de árbol de decisión
       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X, y)

       # Realiza predicciones para diferentes valores de PBrazo (cm) y PCB (mm)
       pbrazo_values = np.linspace(min(X['PBrazo (cm)']), max(X['PBrazo (cm)']), 100)
       pcb_values = np.linspace(min(X['PCB (mm)']), max(X['PCB (mm)']), 100)
       pbrazo_values, pcb_values = np.meshgrid(pbrazo_values, pcb_values)
       grasa_pred_dt = modelo_grasa_dt.predict(np.column_stack((pbrazo_values.ravel(), pcb_values.ravel())))

       # Calcula el coeficiente de determinación (R^2) para el modelo
       r2_grasa_dt = modelo_grasa_dt.score(X, y)

       # Crear una figura 3D para visualizar los datos y las predicciones del modelo
       fig = plt.figure(figsize=(12, 6))
       ax = fig.add_subplot(111, projection='3d')
       
       # Grafica los datos reales
       scatter = ax.scatter(X['PBrazo (cm)'], X['PCB (mm)'], y, label='Datos reales', c='blue')

       # Grafica las predicciones del modelo
       ax.plot_surface(pbrazo_values, pcb_values, grasa_pred_dt.reshape(pbrazo_values.shape), alpha=0.5, color='green')

       # Etiquetas de los ejes
       ax.set_xlabel('PBrazo (cm)')
       ax.set_ylabel('PCB (mm)')
       ax.set_zlabel('Grasa Corporal (%)')

       # Crear una leyenda ficticia para el gráfico
       legend = ax.legend(*scatter.legend_elements(), title="Datos reales")
       ax.add_artist(legend)

       # Título del gráfico
       plt.title(f'Grasa Corporal en función de PBrazo (cm) y PCB (mm) (R^2={r2_grasa_dt:.2f})')

       # Mostrar el gráfico en Streamlit
       st.pyplot(fig)

       #################

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.tree import DecisionTreeRegressor

       # Carga tus datos y define tus características (X) y la variable dependiente (y)
       X = data[['PBrazo (cm)', 'PCB (mm)']]
       y = data['Grasa Corporal (%)']

       # Entrena un modelo de árbol de decisión       
       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X, y)

       # Define rangos de valores para PBrazo (cm) y PCB (mm)
       pbrazo_range = np.linspace(X['PBrazo (cm)'].min(), X['PBrazo (cm)'].max(), 100)
       pcb_range = np.linspace(X['PCB (mm)'].min(), X['PCB (mm)'].max(), 100)
       pbrazo_grid, pcb_grid = np.meshgrid(pbrazo_range, pcb_range)

       # Combina las características en una matriz bidimensional
       X_grid = np.c_[pbrazo_grid.ravel(), pcb_grid.ravel()]

       # Realiza predicciones en la malla de valores
       y_pred = modelo_grasa_dt.predict(X_grid)
       y_pred = y_pred.reshape(pbrazo_grid.shape)

       # Crea la gráfica de superficie de decisión
       figu = plt.figure(figsize=(10, 6))
       #ax = figu.add_subplot(111, projection='3d')
       contour = plt.contourf(pbrazo_grid, pcb_grid, y_pred, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
       plt.scatter(X['PBrazo (cm)'], X['PCB (mm)'], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
       plt.xlabel('PBrazo (cm)')
       plt.ylabel('PCB (mm)')
       #plt.title('Grasa Corporal (%)')
       cbar = plt.colorbar(contour)
       cbar.set_label('Grasa Corporal (%)')

       # Título del gráfico
       plt.title(f'Superficie de Decisión del Árbol de Decisión para Grasa Corporal (%) (R^2={r2_grasa_dt:.2f})')

       # Mostrar el gráfico en Streamlit
       st.pyplot(figu)

       ###################3

       import streamlit as st
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor
       import matplotlib.pyplot as plt
       from io import StringIO

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PBrazo (cm)', 'PCB (mm)']]
       y = data['Grasa Corporal (%)']

       # Crea un modelo de árbol de decisión
       modelo_grasa_dt = DecisionTreeRegressor(max_depth=4)
       modelo_grasa_dt.fit(X, y)

       # Genera el diagrama del árbol de decisión       
       plt.figure(figsize=(70, 25))
       from sklearn.tree import plot_tree
       plot_tree(modelo_grasa_dt, filled=True, feature_names=X.columns, fontsize=20)
       plt.title("Árbol de Decisión para Grasa Corporal (%) vs. PBrazo (cm) y PCB (mm)", fontsize=24)

       # Mostrar el diagrama del árbol en Streamlit
       st.pyplot(plt)

       ################

       import streamlit as st
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.tree import export_text

       # Programa para que el usuario ingrese valores y obtenga una predicción
       def predecir_grasa_corporal(pbrazo, pcb):
           # Crea un modelo de árbol de decisión
           modelo_grasa_dt = DecisionTreeRegressor(max_depth=4)
           modelo_grasa_dt.fit(X, y)

           # Realiza una predicción utilizando el modelo de árbol de decisión
           predicción = modelo_grasa_dt.predict([[pbrazo, pcb]])[0]

           return predicción

       # Carga tus datos desde reduced_df_2 (reemplaza 'data.csv' con tu propio DataFrame)
       # Asumiendo que los datos están en formato CSV

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PBrazo (cm)', 'PCB (mm)']]
       y = data['Grasa Corporal (%)']

       # Extraer las reglas de decisión
       tree_rules = export_text(modelo_grasa_dt, feature_names=list(X.columns))

       # Mostrar las reglas de decisión en Streamlit
       st.text("Reglas de Decisión:")
       st.text(tree_rules)

       # Interfaz de usuario para ingresar valores y obtener predicciones
       #st.sidebar.header("Ingrese los valores para la predicción:")
       #pbrazo_input = st.sidebar.number_input("PBrazo (cm):", min_value=0.0)
       #pcb_input = st.sidebar.number_input("PCB (mm):", min_value=0.0)
       pbrazo_input = st.number_input("Ingresa el valor del perimetro de brazo (cm): ")
       pcb_input = st.number_input("Ingresa el valor del pliegue subcutaneo escapular (mm): ")


#if st.button("Predicción"):
#       predicción = predecir_grasa_corporal(pbrazo_input, pcb_input)
#       st.write(f"Predicción de Grasa Corporal (%): {predicción:.2f}")


       # Usando la opción 'key' para el botón para asegurar que se active después de que se ingresen los valores
       if st.button("Predicción", key="predicto_button"):
           predicción = predecir_grasa_corporal(pbrazo_input, pcb_input)
           st.write(f"Predicción de Grasa Corporal (%): {predicción:.2f}")

       ###################


# Contenido de la pestaña 3
elif pestañas == "Predicción de Sarcopenia":
       st.subheader("Contenido de la Pestaña 3")       
       st.write("Aquí puedes agregar contenido para la tercera pestaña.")

       #import streamlit as st       
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
       N_df = df[['Folio', 'Músculo (kg)', 'Grasa Corporal (%)', 'FA', 'Velocidad de marcha']]

       def clasificar_filas(df):
           clasificaciones = []
           for _, fila in df.iterrows():
               if fila['FA'] <= 23.90:
                   if fila['Músculo (kg)'] <= 62.81:
                       if fila['Grasa Corporal (%)'] <= 43.65:
                           if fila['Velocidad de marcha'] <= 0.55:
                               clasificacion = 3.0
                           else:
                               if fila['Velocidad de marcha'] <= 0.75:
                                   clasificacion = 1.0
                               else:
                                   clasificacion = 1.0
                       else:
                           clasificacion = 3.0
                   else:
                       clasificacion = 0.0
               else:
                   if fila['FA'] <= 32.60:
                       if fila['Músculo (kg)'] <= 61.80:
                           clasificacion = 2.0
                       else:
                           clasificacion = 0.0
                   else:
                       clasificacion = 2.0
               clasificaciones.append(clasificacion)
           df["Clasificación"] = clasificaciones
           return df

# Carga tus datos desde reduced_df_2 (reemplaza 'data.csv' con tu propio DataFrame)
# Asumiendo que los datos están en formato CSV

# Utiliza la función para clasificar las filas de tu DataFrame
# Reemplaza 'N_df' con tu DataFrame
       clasificado_df = clasificar_filas(N_df.copy())

# Ahora, en el DataFrame original, tendrás una nueva columna llamada "Clasificación" con las clasificaciones correspondientes.
       st.dataframe(clasificado_df)

##########################3

       import streamlit as st
       import matplotlib.pyplot as plt

       # Contar la cantidad de pacientes en cada clasificación
       clasificacion_counts = clasificado_df['Clasificación'].value_counts()

       # Obtener las etiquetas de las clasificaciones y sus valores
       etiquetas = clasificacion_counts.index
       valores = clasificacion_counts.values

       # Crear un gráfico de pastel
       fig = plt.figure(figsize=(8, 8))
       plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=140)
       plt.title('Distribución de Clasificaciones')

       # Mostrar el gráfico en Streamlit
       st.pyplot(fig)

       ###############


       import streamlit as st
       import plotly.express as px
       import plotly.graph_objects as go
       from plotly.subplots import make_subplots

       # Obtener las columnas numéricas
       numeric_columns = clasificado_df.select_dtypes(include='number').columns

       # Calcular el número de filas y columnas del panel
       num_rows = len(numeric_columns)

       # Filtrar el DataFrame para cada clasificación y crear gráficos de caja
       for clasificacion in clasificado_df['Clasificación'].unique():
           df_filtrado = clasificado_df[clasificado_df['Clasificación'] == clasificacion]
           fig = make_subplots(rows=1, cols=len(numeric_columns), shared_yaxes=True, subplot_titles=numeric_columns)
           for i, column in enumerate(numeric_columns):
               box = px.box(df_filtrado, x='Clasificación', y=column, title=column)
               fig.add_trace(box['data'][0], row=1, col=i + 1)
           fig.update_layout(title=f'Clasificación {clasificacion}')
           st.plotly_chart(fig)

######################3

       import streamlit as st
       import plotly.express as px

       # Obtener las columnas numéricas
       numeric_columns = clasificado_df.select_dtypes(include='number').columns

       # Obtener las clasificaciones únicas
       clasificaciones_unicas = clasificado_df['Clasificación'].unique()

       # Filtrar el DataFrame para cada parámetro y crear un único gráfico de caja para cada uno
       for column in numeric_columns:
           fig = px.box(clasificado_df, x='Clasificación', y=column, title=column, notched=True, points='all')
           st.plotly_chart(fig)

       ############################


       import streamlit as st
       import pandas as pd

       # Función para calcular el músculo
       def calcular_musculo(fa, ppantorrilla):
           if fa <= 27.90:
               if ppantorrilla <= 34.55:
                   if fa <= 15.77:
                       if ppantorrilla <= 32.00:
                           return 35.42
                       else:
                           return 37.38
                   else:
                       if fa <= 16.38:
                           return 58.30
                       else:
                           return 37.63
               else:
                   if fa <= 15.25:
                       return 51.90
                   else:
                       if ppantorrilla <= 41.25:
                           return 40.96
                       else:
                           return 50.20
           else:
               if fa <= 32.80:
                   if ppantorrilla <= 36.45:
                       if ppantorrilla <= 34.95:
                           return 52.20
                       else:
                           return 52.70
                   else:
                       return 47.10
               else:
                   if ppantorrilla <= 36.75:
                       return 54.20
                   else:
                       if fa <= 36.27:
                           return 61.10
                       else:
                           return 60.00

       # Función para calcular la grasa corporal
       def calcular_grasa(pbrazo, pcb):
           if pcb <= 9.50:
               if pbrazo <= 27.65:
                   if pbrazo <= 24.75:
                       if pcb <= 6.50:
                           return 26.50
                       else:
                           return 26.40
                   else:
                       if pcb <= 6.50:
                           return 34.70
                       else:
                           return 30.37
               else:
                   if pcb <= 7.50:
                       if pbrazo <= 30.75:
                           return 20.60
                       else:
                           return 27.07
                   else:
                       if pbrazo <= 29.15:
                           return 27.90
                       else:
                           return 30.80
           else:
               if pbrazo <= 28.75:
                   if pcb <= 11.00:
                       if pbrazo <= 27.65:
                           return 35.40
                       else:
                           return 34.50
                   else:
                       if pbrazo <= 25.85:
                           return 31.50
                       else:
                           return 28.75
               else:
                   if pbrazo <= 35.25:
                       if pcb <= 18.50:
                           return 37.19
                       else:
                           return 30.60
                   else:
                       if pcb <= 19.00:
                           return 44.70
                       else:
                           return 37.60

       # Aplicar las funciones a las columnas correspondientes de df
       df['Musculo_pred (kg)'] = df.apply(lambda row: calcular_musculo(row['FA'], row['PPantorrilla (cm)']), axis=1)
       df['Grasa Corporal_pred (%)'] = df.apply(lambda row: calcular_grasa(row['PBrazo (cm)'], row['PCB (mm)']), axis=1)

       # Guardar el DataFrame actualizado en un archivo Excel
       nombre_archivo_excel = "df_con_predicciones.xlsx"
       df.to_excel(nombre_archivo_excel, index=False)
       st.write(f"Se ha guardado el DataFrame actualizado en {nombre_archivo_excel}")
       df_2
       clasificar_filas(df_2)
       df_2

       import streamlit as st
       import pandas as pd
       import numpy as np
       from sklearn.preprocessing import StandardScaler
       from sklearn.cluster import AgglomerativeClustering

       N_df_filtro = df_2[df_2['Clasificación'] == 1.0]
       # Cargar los datos en un dataframe
       # Elimina la columna "Clasificación" del DataFrame N_df_filtro
       N_df_filtro = N_df_filtro.drop(columns=['Clasificación'])
       data_2023 = N_df_filtro

       # Seleccionar solo las columnas numéricas
       numeric_data_2023 = data_2023.select_dtypes(include='number')

       # Eliminar valores no numéricos
       numeric_data_2023 = numeric_data_2023.dropna()

       # Normalizar los datos
       scaler = StandardScaler()
       normalized_data_2023 = scaler.fit_transform(numeric_data_2023)
       
       # Obtener los valores máximos y mínimos de las columnas originales
       min_values = np.min(numeric_data_2023, axis=0)
       max_values = np.max(numeric_data_2023, axis=0)

       # Aplicar hierarchical clustering
       clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
       labels_2023 = clustering.fit_predict(normalized_data_2023)

       # Agregar las etiquetas al dataframe original
       data_2023['Cluster'] = labels_2023

       # Guardar el dataframe con las etiquetas en un archivo Excel
       nombre_archivo_excel = "data_2023_clustered.xlsx"
       data_2023.to_excel(nombre_archivo_excel, index=False)
       st.write(f"Se ha guardado el DataFrame con etiquetas en {nombre_archivo_excel}")

       # Filtrar el DataFrame para la clasificación 1.0
       df_clasif_0 = data_2023[data_2023['Cluster'] == 0.0]

       # Filtrar el DataFrame para la clasificación 2.0
       df_clasif_1 = data_2023[data_2023['Cluster'] == 1.0]

       # Filtrar el DataFrame para la clasificación 2.0
       df_clasif_2 = data_2023[data_2023['Cluster'] == 2.0]

       # Guardar los DataFrames en archivos Excel
       df_clasif_0.to_excel('clasificacion_0.xlsx', index=False)
       df_clasif_1.to_excel('clasificacion_1.xlsx', index=False)
       df_clasif_2.to_excel('clasificacion_2.xlsx', index=False)

       import streamlit as st
       import pandas as pd
       import plotly.express as px

       # Obtener las columnas numéricas
       numeric_columns = data_2023.select_dtypes(include='number').columns

       # Obtener las clasificaciones únicas
       clasificaciones_unicas = data_2023['Cluster'].unique()

       # Calcular el número de filas y columnas del panel
       num_rows = len(numeric_columns)

       # Filtrar el DataFrame para cada parámetro y crear un único gráfico de caja para cada uno
       for column in numeric_columns:
           fig = px.box(data_2023, x='Cluster', y=column, title=column, notched=True, points='all')
           st.plotly_chart(fig)
else:
       st.subheader("Equipo de Trabajo")

       # Información del equipo
       equipo = [{
               "nombre": "Dr. Santiago Arceo Díaz",
               "foto": "ArceoS.jpg",
               "reseña": "Licenciado en Física, Maestro en Física y Doctor en Ciencias (Astrofísica). Posdoctorante de la Universidad de Colima y profesor del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, pertenece al núcleo académico y es colaborador del cuerpo académico Tecnologías Emergentes y Desarrollo Web de la Maestría Sistemas Computacionales. Ha dirigido tesis de la Maestría en Sistemas Computacionales y en la Maestría en Arquitectura Sostenible y Gestión Urbana.",
               "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "santiagoarceodiaz@gmail.com"},
           {
               "nombre": "José Ramón González",
               "foto": "JR.jpeg",
               "reseña": "Estudiante de la facultad de medicina en la Universidad de Colima, cursando el servicio social en investigación en el Centro Universitario de Investigaciones Biomédicas, bajo el proyecto Aplicación de un software basado en modelos predictivos como herramienta de apoyo en el diagnóstico de sarcopenia en personas adultas mayores a partir de parámetros antropométricos.", "CV": "https://scholar.google.com.mx/citations?user=3xPPTLoAAAAJ&hl=es", "contacto": "jgonzalez90@ucol.mx"},
           {
               "nombre": "Dra. Xochitl Angélica Rosío Trujillo Trujillo",
               "foto": "DraXochilt.jpg",
               "reseña": "Bióloga, Maestra y Doctora en Ciencias Fisiológicas con especialidad en Fisiología. Es Profesora-Investigadora de Tiempo Completo de la Universidad de Colima. Cuenta con perfil deseable y es miembro del Sistema Nacional de Investigadores en el nivel 3. Su línea de investigación es en Biomedicina en la que cuenta con una producción científica de más de noventa artículos en revistas internacionales, varios capítulos de libro y dos libros. Imparte docencia y ha formado a más de treinta estudiantes de licenciatura y de posgrado en programas académicos adscritos al Sistema Nacional de Posgrado del CONAHCYT.",
               "CV": "https://portal.ucol.mx/cuib/XochitlTrujillo.htm", "contacto": "rosio@ucol.mx"},
                 {
               "nombre": "Dr. Miguel Huerta Viera",
               "foto": "DrHuerta.jpg",
               "reseña": "Doctor en Ciencias con especialidad en Fisiología y Biofísica. Es Profesor-Investigador Titular “C” del Centro Universitario de Investigaciones Biomédicas de la Universidad de Colima. Es miembro del Sistema Nacional de Investigadores en el nivel 3 emérito. Su campo de investigación es la Biomedicina, con énfasis en la fisiología y biofísica del sistema neuromuscular y la fisiopatología de la diabetes mellitus. Ha publicado más de cien artículos revistas indizadas al Journal of Citation Reports y ha graduado a más de 40 Maestros y Doctores en Ciencias en programas SNP-CONAHCyT.",
               "CV": "https://portal.ucol.mx/cuib/dr-miguel-huerta.htm", "contacto": "huertam@ucol.mx"},
                 {
               "nombre": "Dr. Jaime Alberto Bricio Barrios",
               "foto":  "BricioJ.jpg",
               "reseña": "Licenciado en Nutrición, Maestro en Ciencias Médicas, Maestro en Seguridad Alimentaria y Doctor en Ciencias Médicas. Profesor e Investigador de Tiempo Completo de la Facultad de Medicina en la Universidad de Colima. miembro del Sistema Nacional de Investigadores en el nivel 1. Miembro fundador de la asociación civil DAYIN (Desarrollo de Ayuda con Investigación)",
               "CV": "https://scholar.google.com.mx/citations?hl=es&user=ugl-bksAAAAJ", "contacto": "jbricio@ucol.mx"},      
               {
               "nombre": "Mtra. Elena Elsa Bricio Barrios",
               "foto": "BricioE.jpg",
               "reseña": "Química Metalúrgica, Maestra en Ciencias en Ingeniería Química y doctorante en Ingeniería Química. Actualmente es profesora del Tecnológico Nacional de México Campus Colima. Cuenta con el perfil deseable, es miembro del cuerpo académico Tecnologías Emergentes y Desarrollo Web y ha codirigido tesis de la Maestría en Sistemas Computacionales.",
               "CV": "https://scholar.google.com.mx/citations?hl=es&user=TGZGewEAAAAJ", "contacto": "elena.bricio@colima.tecnm.mx"}
       ]

       # Establecer la altura deseada para las imágenes
       altura_imagen = 150  # Cambia este valor según tus preferencias

       # Mostrar información de cada miembro del equipo
       for miembro in equipo:
           st.subheader(miembro["nombre"])
           img = st.image(miembro["foto"], caption=f"Foto de {miembro['nombre']}", use_column_width=False, width=altura_imagen)
           st.write(f"Correo electrónico: {miembro['contacto']}")
           st.write(f"Reseña profesional: {miembro['reseña']}")
           st.write(f"CV: {miembro['CV']}")

       # Información de contacto
       st.subheader("Información de Contacto")
       st.write("Si deseas ponerte en contacto con nuestro equipo, puedes enviar un correo a santiagoarceodiaz@gmail.com")
