
#%pprint
import pandas as pd #importa la paquetería PAnel DAta (llamada pandas)
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted # importa paqueteria para graficar diagramas de venn
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt #importa pyplot para hacer gáficas
import numpy as np #importar numpy
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
    page_title="AntropoFit",
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
pestañas = st.sidebar.radio("Selecciona una pestaña:", ("Presentación", "Modelos con una variable", "Modelos con 2 variables", "Predicción de Sarcopenia", "Calculadora",  "Equipo de trabajo"))
if pestañas == "Presentación":
       st.title("Sobre AntropoFit")
       st.markdown("""
       Esta aplicación es resultado del proyecto de estancia posdoctoral "**Identificación 
       de las etapas y tipos de sarcopenia mediante modelos predictivos como herramienta 
       de apoyo en el diagnóstico a partir de parámetros antropométricos**", desarrollado 
       por el Doctor en Ciencias (Astrofísica) Santiago Arceo Díaz, bajo la dirección de 
       la Doctora Xóchitl Rosío Angélica Trujillo Trujillo, y con la ayuda de los colaboradores mencionados en esta sección. Esta estancia es gracias a la 
       colaboración entre el entre el **Consejo Nacional de Humanidades Ciencia y Tecnología ([**CONAHCYT**](https://conahcyt.mx/)) y la Universidad de Colima ([**UCOL**](https://portal.ucol.mx/cuib/))**
       """)
       st.subheader("Muestra")
       st.markdown("""
       Los datos utilizados para los modelos se recolectaron a partir de un grupo de voluntarios de centros 
       de convivencia de personas adultas mayores, residentes en las Ciudades de Colima y Villa de Álvarez.
       **En la presente aplicación se crean modelos que permiten estimar variables como el porcentaje de grasa 
       corporal y masa muscular en personas adultas mayores**, permitiendo la evaluación de síndromes geriátricos 
       como la sarcopenia en situaciones en las que no se cuente con el equipo de medición adecuado.
       """)

       st.subheader("Algoritmos y lenguaje de programación")
       st.markdown("""

       Elegimos el lenguaje de programación [**Python**](https://docs.python.org/es/3/tutorial/) y las plataformas [**Streamlit**](https://streamlit.io/) y [**GitHub**](https://github.com/). Estas opciones permiten una fácil visualización y manipulación de la aplicación, además de almacenar los algoritmos en la nube. Las técnicas utilizadas para el análisis de los datos y la creación de modelos de aproximación se derivan de prácticas usuales para la depuración de datos, la creación de árboles de ajuste, la técnica de clustering jerárquico y Random Forest. **La aplicación es de libre acceso y uso gratuito para cualquier personal de atención primaria de pacientes geriátricos.**
       """         
               )

       #st.title("Acerca de Sarc-open-IA")

       st.subheader("Objetivo")
       st.markdown("""
       El objetivo de esta aplicación es asistir al personal médico en la captura, almacenamiento 
       y análisis de datos antropométricos de adultos mayores para la determinación de dependencia 
       funcional y sarcopenia. 
       """)

       st.subheader("Ventajas y características")

       st.markdown("""

            - **Facilitar uso:** Queríamos que nuestra herramienta fuera fácil de usar para el personal médico, incluso si no estaban familiarizados con la inteligencia artificial o la programación. Para lograrlo, elegimos el lenguaje de programación [**Python**](https://docs.python.org/es/3/tutorial/) y las plataformas [**Streamlit**](https://streamlit.io/) y [**GitHub**](https://github.com/). Estas opciones permiten una fácil visualización y manipulación de la aplicación, además de almacenar los algoritmos en la nube.

            - **Interfaz amigable:** El resultado es una interfaz gráfica que permite a los médicos ingresar los datos antropométricos de los pacientes y ver gráficas útiles para el análisis estadístico. También ofrece un diagnóstico en tiempo real de la sarcopenia, y todo esto se hace utilizando cajas de texto y deslizadores para ingresar y manipular los datos.

            - **Accesibilidad total:** El personal médico puede descargar de forma segura las gráficas y los archivos generados por la aplicación. Además, pueden acceder a ella desde cualquier dispositivo con conexión a internet, ya sea un teléfono celular, una computadora, tablet o laptop.
        """)

       st.subheader("Método")
       st.markdown("""
       A partir de datos registrados entre septiembre y octubre el año 2023 en una muestra de adultos mayores que residen en la Zona Metropolitana, Colima, Villa de Álvarez, México, se procedió al desarrollo de modelos predictivos mediante el algoritmo [**Random Forest**](https://cienciadedatos.net/documentos/py08_random_forest_python). En este caso, se crearon modelos que permiten estimar la [**masa muscular**](https://www.scielo.cl/scielo.php?pid=S0717-75182008000400003&script=sci_arttext&tlng=en) (medida en kilogramos) y el [**porcentaje corporal de grasa**](https://ve.scielo.org/scielo.php?pid=S0004-06222007000400008&script=sci_arttext) a partir de distintas medidas antropométricas. 
       
       Los modelos generados muestran un grado aceptable de coincidencia con las mediciones de estos parámetros, que típicamente requieren de balanzas de bioimpedancia y/o absorciometría de rayos X de energía dual. Una vez con las aproximaciones para masa muscular y porcentaje de grasa corporal, se estima el grado de riesgo de padecer sarcopenia para cada paciente mediante el uso del algoritmo de clustering jerarquico. 
       
       Estas condiciones de diagnóstico fueron propuestas con el objetivo de minimizar la cantidad de parámetros antropométricos y establecer puntos de corte que puedan ser validados por personal médico capacitado. **Este enfoque se asemeja a lo que se conoce en inteligencia artificial como un sistema experto, ya que los modelos resultantes requieren validación por parte de especialistas.**
       """
       )




# Contenido de la pestaña 1
if pestañas == "Modelos con una variable":
       st.title("Modelos de aproximación de una sola variable independiente")
       st.markdown("""
       En esta pestaña se muestra el proceso para calcular **modelos de aproximación** a la masa muscular, medida en kilogramos, y el porcentaje de grasa corporal **a partir de una sola variable**. En el caso de la masa muscular, se predicen valores para pacientes  a partir del [**perímetro de pantorrilla**](https://scielo.isciii.es/pdf/nh/v33n3/10_original9.pdf) y en el caso de la grasa corporal se utiliza el [**perímetro de brazo**](https://www.sciencedirect.com/science/article/pii/S0212656709006416).
       """)
       st.markdown("""
       A continuación se muestra la base de datos de adultos mayores. En la parte superior de cada columna se muestra el nombre del parámetro y las unidades correspondientes. Si deja el ícono del mouse en la parte superior derecha puede descargar la tabla con los datos.
       """)

       with st.expander("**Claves de variables**"):
           st.markdown("""
           - **Folio:** identificador personal.
           - **Peso (kg):** peso corporal total, medida en Kg.
           - **Talla (cm):** altura de la persona, medida en centímetros.
           - **IMC:** índice de masa corporal.
           - **PCintura (cm):** perímetro de cintura, medido en centímetros.
           - **Pcadera (cm):** perímetro de la cadera, medido en centímetros.
           - **PBrazo (cm):** perímetro total del brazo, medido en centímetros. 
           - **PPantorrilla (cm):** perímetro de la pantorrilla, medida en centímetros.
           - **PCB (mm):** pliegue cutáneo de brazo, medido en milímetros.
           - **PCT (mm):** pliegue cutáneo del triceps, medido en milímetros.
           - **PCSE (mm):** pliegue cutáneo sub escapular, medido en milímetros.
           - **Agua corporal (%):** porcentaje corporal de agua.
           - **Músculo (kg):** peso del corporal.
           - **Grasa corporal (%):** porcentaje corporal de grasa.
           - **Gs Brazo derecho:** porcentaje de grasa subcutánea en el brazo derecho. 
           - **Gs Brazo izquierdo:** porcentaje de grasa subcutánea en el brazo izquierdo.
           - **Gs pierna izquierda:** porcentaje de grasa subcutánea en la pierna izquierda.
           - **Gs pierna derecha:** porcentaje de grasa subctánea en la pierna derecha.
           - **Centro:** porcentaje de grasa en la zona central del abdomen.
           - **Fuerza mano derecha:** Fuerza de agarre en la mano derecha, medida en kg de presión.
           - **Fuerza mano izquierda:** Fuerza de agarre en la mano izquierda, medida en kg de presión.
           - **Velocidad de marcha:** velocidad de marcha, medida en metros sobre segundo.
           - **PA sistólica:** Presión sanguínea sistólica, medida en milímetros de mercurio.
           - **PA diastólica:** Presión sanguínea diastólica, medida en milímetros de mercurio.
           """)

    
        
       #import streamlit as st       
       df=pd.read_excel('AM_2023_Antropo.xlsx')
       st.dataframe(df)
       with st.expander("**Información adicional**"):
           # Mostrar información adicional sobre el DataFrame
           num_rows, num_columns = df.shape
           missing_data = df.isnull().any().any()

           st.write(f"**Número de filas**: {num_rows}")
           st.write(f"**Número de columnas**: {num_columns}")
           if missing_data:
               st.write("Existen datos faltantes en alguna fila.")
           else:
               st.write("No hay datos faltantes en ninguna fila.")

       # Crear un botón de descarga para el dataframe
       def download_button(df, filename, button_text):
           # Crear un objeto ExcelWriter
           excel_writer = pd.ExcelWriter(filename, engine='xlsxwriter')
           # Guardar el dataframe en el objeto ExcelWriter
           df.to_excel(excel_writer, index=False)
           # Cerrar el objeto ExcelWriter
           excel_writer.save()
           # Leer el archivo guardado como bytes
           with open(filename, 'rb') as f:
               file_bytes = f.read()
               # Generar el enlace de descarga
               href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_bytes).decode()}" download="{filename}">{button_text}</a>'
               st.markdown(href, unsafe_allow_html=True)

       # Crear un botón de descarga para el dataframe
       def download_button_CSV(df, filename, button_text):
           csv = df.to_csv(index=False)
           b64 = base64.b64encode(csv.encode()).decode()
           href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_text}</a>'
           st.markdown(href, unsafe_allow_html=True)


       # Dividir la página en dos columnas
       col1, col2 = st.columns(2)
       # Agregar un botón de descarga para el dataframe en la primera columna
       with col1:
           download_button(df, 'muestra_antrompométrica_colima_2023.xlsx', 'Descargar como Excel')
           st.write('')
       # Agregar un botón de descarga para el dataframe en la segunda columna
       with col2:
           download_button_CSV(df, 'muestra_antrompométrica_colima_2023.csv', 'Descargar como CSV')
           st.write('')
    
       st.subheader("Modelos predictivos basados en árboles de regresión")

       st.markdown("""
      En esta sección presetamos los modelos predictivos para masa muscular y porcentaje de grasa corporal hechos a partir de [**árboles de regresión**](https://www.researchgate.net/publication/242370834_Classification_and_Regression_Trees_An_Introduction). 
      """)

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
       st.subheader('Modelos de Regresión para Predicción de Músculo')

       # Muestra una tabla con los primeros registros de los datos
       st.markdown("""
       Esta es la base de datos con parámetros antropométricos:
       """)       #st.write(data.head())
       st.dataframe(data)
       with st.expander("**Información adicional**"):
           # Mostrar información adicional sobre el DataFrame
           num_rows, num_columns = data.shape
           missing_data = data.isnull().any().any()

           st.write(f"**Número de filas**: {num_rows}")
           st.write(f"**Número de columnas**: {num_columns}")
           if missing_data:
               st.write("Existen datos faltantes en alguna fila.")
           else:
               st.write("No hay datos faltantes en ninguna fila.")


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
       st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_musculo_lr}, **Intercepto** = {intercepto_musculo_lr}')

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

       # Generar el diagrama del árbol de decisión simplificado con un tamaño de letra más grande
       figur = plt.figure(figsize=(40, 15), dpi=600)
       plt.rc('font', size=12)  # Ajusta el tamaño de fuente aquí
       plot_tree(modelo_musculo_dt_simplified, filled=True, feature_names=X.columns, fontsize=12)  # Ajusta el tamaño de la letra aquí
       plt.title("Árbol de Decisión Simplificado para Musculo (kg) vs. PPantorrilla (cm)", fontsize=30)  # Ajusta el tamaño de fuente del título aquí

       # Mostrar la figura en Streamlit
       st.pyplot(figur)

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
       st.write("En esta gráfica se comparan los modelos con los datos medidos (puntos azule). Las curvas de distintos colores correponden a: modelo lineal (en rojo), aproximación de Random Forest (azul) y aproximación de árbol de decisión (verde)")
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
       st.title("Modelos de aproximación con dos variables independientes")
       st.write("Se muestran los modelos en los que se estima grasa corporal y mas muscular usando pares de variables: perímetro de pantorrilla y fuerza de agarre, en el caso de masa muscular, y perímetro de brazo y pliegue cutáneo subescapular, en el caso del porcentaje de grasa corporal")
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
       st.title("Estimación de sarcopenia")       
       st.write("Se usan los resultados de los modelos aproximados y el algoritmo de clustering jerárquico para estimar sarcopenia")

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
           
elif pestañas == "Calculadora":
    st.title("Calculadora")
    import streamlit as st
    import pandas as pd
    st.markdown("""
    A continuación pude cargar un archivo para la clasificación de pacientes (el formato puede ser .xlsx o.csv). Para que la calculadora puede crear aproximaciones la masa muscular, el porcentaje de grasa corporal y riesgo de sarcopenia, necesita que en su archivo se cuente con las variables:
    - **Perímetro de brazo**, medido en centímetros (el nombre de la columna debe ser "**PBrazo (cm)**").
    - **Perímetro de pantorrilla**, medido en centímetros (el nombre de columna debe ser "**PPantorrilla (cm)**").
    - **Fuerza de agarre**, medida en kilogramos (el nombre debe ser "**FA**").
    - **Pliegue cutáneo de brazo**, medido en milímetros (el nombre de la columna debe ser "**PCB (mm)**")
    """)
    # Función para cargar un archivo
    def cargar_archivo():
        uploaded_file = st.sidebar.file_uploader("Cargar archivo", type=["xlsx", "xls", "csv"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    # Leer el archivo CSV en un DataFrame de pandas
                    df = pd.read_csv(uploaded_file)
                else:
                    # Leer el archivo Excel en un DataFrame de pandas
                    df = pd.read_excel(uploaded_file)
                st.write("¡Archivo cargado correctamente!")
                return df  # Devolver el DataFrame cargado
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

    # Llamar a la función para cargar el archivo
    df = cargar_archivo()
    st.write("**Esta es su tabla:**")
    # Mostrar el DataFrame en el área principal de la aplicación
    if df is not None:
        st.dataframe(df)  # Mostrar el DataFrame cargado en el área principal

        # Crear una barra lateral para las pestañas
        pestañas = st.sidebar.radio("Selecciona una pestaña:", ("Calcular con una variable", "Calcular con dos variables"))
        if pestañas == "Calcular con una variable":
               st.markdown("""
               La calculadora utilizará dos algoritmos de **"arboles de ajuste"** que calcularán aproximaciones para la masa 
               muscular total del paciente y el porcentaje de grasa corporal a partir de las variables de interés (Perímetro de 
               pantorrilla y fuerza de agarre para el cálculo de la masa muscular y pliegue cutáneo de brazo y perímetro de brazo 
               para el porcentaje de grasa corporal. En la tabla que puede obervar abajo econtrará dos coulumnas en donde se muestran 
               estas aproximaciones: "Musculo_pred (Kg)" y "Grasa Corporal_pred (%)"). Además, encontrará una tercer columna (llamada 
               **"Clasificación"** en la que aquellos pacientes con el mayor grado de simulitud serán agrupados en tres conjuntos).
               """)
               import streamlit as st       
               import pandas as pd
               from sklearn.tree import DecisionTreeRegressor
               import numpy as np
               import matplotlib.pyplot as plt
               from mpl_toolkits.mplot3d import Axes3D
               from sklearn.tree import DecisionTreeRegressor, export_text


               #st.dataframe(df)
               df=df.dropna()
               df['FA'] = (df['Fuerza mano derecha'] + df['Fuerza mano izquierda']) / 2
               df['Gs Brazo'] = (df['Gs Brazo derecho'] + df['Gs Brazo izquierdo']) / 2
               df['Gs Pierna'] = (df['Gs pierna derecha'] + df['Gs pierna izquierda']) / 2
               df=df[['Folio', 'Peso (kg)', 'Talla (cm)', 'IMC', 'PCintura (cm)',
                      'PCadera (cm)', 'PBrazo (cm)', 'PPantorrilla (cm)', 'PCB (mm)',
                      'PCT (mm)', 'PCSE (mm)', 'Agua Corporal (%)', 'Músculo (kg)',
                      'Grasa Corporal (%)', 'Centro',
                      'FA','Velocidad de marcha']]

            
               data=df
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
               #st.write(f"Se ha guardado el DataFrame actualizado en {nombre_archivo_excel}")
               df_2=df
               

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


               clasificado_df = clasificar_filas(df.copy())

               st.dataframe(clasificado_df)

            
               import streamlit as st
               import matplotlib.pyplot as plt

               # Contar la cantidad de pacientes en cada clasificación
               clasificacion_counts = clasificado_df['Clasificación'].value_counts()

               # Obtener las etiquetas de las clasificaciones y sus valores
               etiquetas = clasificacion_counts.index
               valores = clasificacion_counts.values

               # Crear un gráfico de pastel
               fig = plt.figure(figsize=(2, 2))
               plt.pie(valores, labels=etiquetas, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
               plt.title('Distribución de Clasificaciones', fontsize=12)

               # Mostrar el gráfico en Streamlit
               st.pyplot(fig)

               import streamlit as st
               import pandas as pd
               import matplotlib.pyplot as plt
               import seaborn as sns

               # Obtener las columnas numéricas
               columnas_numericas = clasificado_df.select_dtypes(include=['int', 'float']).columns.tolist()

               # Interfaz de usuario
               st.title('Visualización de Datos por Clasificación')
               st.markdown("""
               A continuación puede comparar las caráctristicas de los pacientes que quedaron agrupados en los 
               diferentes conjuntos. La idea detrás de esto es establecer si existen diferencias entre las características 
               de los grupos de pacientes estudiados, con el objetivo de identificar uno o mas grupos con indicios de riesgo
               de sarcopenia.
               En la casilla de verificación puede seleccionar los parámetros de interés y debajo se mostrarán los diagramas 
               de caja para los pacientes de cada grupo (una variable a la vez). 
               """)
               
               # Lista de selección de columnas numéricas
               columnas_seleccionadas = st.multiselect('Selecciona las columnas numéricas:', columnas_numericas)

               # Verificar si se han seleccionado columnas
               if len(columnas_seleccionadas) > 0:
                  # Crear un arreglo de subplots
                  fig, axs = plt.subplots(nrows=len(columnas_seleccionadas), ncols=1, figsize=(10, 6 * len(columnas_seleccionadas)))
                  # Aplanar los ejes
                  axs = np.ravel(axs)
    
                  # Iterar sobre las columnas seleccionadas
                  for i, columna in enumerate(columnas_seleccionadas):
                      # Obtener los valores únicos de la columna 'Clasificación'
                      valores_clasificacion = clasificado_df['Clasificación'].unique()
        
                      # Iterar sobre los valores de clasificación
                      for valor in valores_clasificacion:
                      # Filtrar el DataFrame por el valor de clasificación
                          df_filtrado = clasificado_df[clasificado_df['Clasificación'] == valor]
            
                          # Crear un gráfico de caja para la columna actual y el valor de clasificación actual
                          #sns.boxplot(x='Clasificación', y=columna, data=df_filtrado, ax=axs[i], label=f'Clasificación {valor}')
                          sns.boxplot(x='Clasificación', y=columna, data=df_filtrado, ax=axs[i], hue='Clasificación')

                      # Añadir título y etiquetas al subplot
                      axs[i].set_title(f'Boxplot de {columna}')
                      axs[i].set_xlabel('Clasificación')
                      axs[i].set_ylabel(columna)
    
                  # Ajustar el diseño de los subplots    
                  plt.tight_layout()
    
                  # Mostrar los subplots en Streamlit
                  st.pyplot(fig)
               else:
                  st.write('Por favor, selecciona al menos una columna numérica.')


               import streamlit as st
               import pandas as pd

               # Obtener los valores únicos de la columna "Clasificación"
               valores_clasificacion = clasificado_df["Clasificación"].unique()

               # Crear una caja de selección para elegir los valores de clasificación
               valor_seleccionado = st.multiselect("Seleccionar valor de clasificación:", valores_clasificacion)

               # Filtrar el DataFrame según el valor seleccionado
               df_filtrado = clasificado_df[clasificado_df["Clasificación"].isin(valor_seleccionado)]
               st.markdown(
               """
               En esta sección puede descargar los datos de todos los pacientes clasificados dentro del mismo grupo (el cual puede seleccionar en la casilla de verificación que viene abajo).
               """
                          )
               # Mostrar el DataFrame filtrado
               st.write("DataFrame filtrado:")
               st.write(df_filtrado)

               # Botón de descarga para el DataFrame filtrado como archivo Excel
               csv_file = df_filtrado.to_csv(index=False)
               b64 = base64.b64encode(csv_file.encode()).decode()  # Codificar el archivo CSV como base64
               href = f'<a href="data:file/csv;base64,{b64}" download="df_filtrado.csv">Descargar CSV</a>'
               st.markdown(href, unsafe_allow_html=True)

               # Botón de descarga para el DataFrame filtrado como archivo Excel
               # excel_file = df_filtrado.to_excel(index=False)
               #b64 = base64.b64encode(excel_file).decode()  # Codificar el archivo Excel como base64
               #href = f'<a href="data:application/octet-stream;base64,{b64}" download="df_filtrado.xlsx">Descargar Excel</a>'
               #st.markdown(href, unsafe_allow_html=True)

               import streamlit as st
               import pandas as pd
               import matplotlib.pyplot as plt

               # Obtener las columnas numéricas del DataFrame
               columnas_numericas = df_filtrado.select_dtypes(include=['int', 'float']).columns.tolist()

               # Interfaz de usuario
               st.title('Histogramas de Columnas Numéricas')
               columna_seleccionada = st.selectbox('Selecciona una columna numérica:', columnas_numericas)

               # Obtener el tamaño de los bines desde un deslizador
               tamanio_bin = st.slider('Tamaño de Bin', min_value=1, max_value=50, value=10)

               # Crear una figura y un conjunto de subplots
               fig, ax = plt.subplots()

               # Dibujar el histograma con el tamaño de los bines especificado
               ax.hist(df_filtrado[columna_seleccionada], bins=tamanio_bin)

               # Agregar etiquetas y título
               ax.set_xlabel(columna_seleccionada)
               ax.set_ylabel('Frecuencia')
               ax.set_title(f'Histograma de {columna_seleccionada}')

               # Mostrar el histograma en Streamlit
               st.pyplot(fig)


        else:
            st.title("Ingreso de datos en tiempo real")
            st.markdown("""
                   En esta sección pued ecargar datos indvidualmente y generar un archivo .xlsx o .csv con la información recolectada
               """
                   )

            # Crear un DataFrame vacío para almacenar los datos de los pacientes
            if 'data' not in st.session_state:
                st.session_state.data = pd.DataFrame(columns=["Folio","Edad (años)","Peso (kg)","Altura (cm)","Grasa (%)","Musculo (kg)","PBrazo (cm)","PPantorrilla (cm)",'FA (kg)',"Marcha (ms-1)"])

               # Título
            st.title('Ingreso manual de datos de pacientes')

               # Crear un formulario para agregar datos de un paciente
            st.markdown("""
            En el siguiente espacio puede ingresar los datos de un paciente en observación. Cada una de las cajas permite teclear los resultados de las mediciones.
               """)
            with st.form('Agregar Paciente'):
                Folio = st.text_input('Nombre del Paciente')
                Edad = st.number_input('Edad (años) ', min_value=0, max_value=150)
                Peso = st.number_input("Peso (kg)", min_value=0.0)
                Altura = st.number_input('Altura (cm)', min_value=0.0)
                Grasa = st.number_input('Grasa (%)', min_value=0.0)
                Musculo = st.number_input('Musculo (kg)', min_value=5.0)
                PBrazo = st.number_input('PBrazo (cm)', min_value=0.0)
                PPantorrilla = st.number_input('PPantorrilla (cm)', min_value=0.0)
                FA = st.number_input('FA (kg)', min_value=0.0)
                Marcha = st.number_input(' Marcha (ms-1)', min_value=0.0)

                if st.form_submit_button('Agregar Paciente'):
                    st.session_state.data = st.session_state.data.append({'Folio': Folio, 'Edad (años)': Edad, 'Peso (kg)': Peso, 'Altura (cm)': Altura, 'Grasa (%)': Grasa, 'Musculo': Musculo, 'PBrazo (cm)': PBrazo, 'PPantorriilla (cm)': PPantorrilla, 'FA (kg)': FA, 'Marcha (ms-1)': Marcha}, ignore_index=True)
                    st.success('Datos del paciente agregados con éxito!')
############
            import streamlit as st
            import pandas as pd
            import io
            import base64
            import pickle
            st.write("En esta sección es posible editar los datos de cualquier paciente previamente registrado. En la caja de ingreso de datos, escriba el número de fila a editar y cambien los valores del campo a modificar. Una vez realizados los cambios, haga clic en el botón de *Guardar cambios*.")
                # Ingresar el número de fila a editar
            edit_row_number = st.number_input('Número de Fila a Editar', min_value=0, max_value=len(st.session_state.data)-1, value=0, step=1, key='edit_row_number')

                # Crear un formulario para editar datos de un paciente
            if edit_row_number is not None:
                with st.form('Editar Paciente'):
                    st.subheader('Editar Fila {}'.format(edit_row_number))
                    data_table = st.session_state.data.copy()
                    st.dataframe(data_table, height=400, width=800)

                    Folio = st.text_input('Nombre del Paciente', value=data_table.iloc[edit_row_number]['Folio'])
                    Edad = st.number_input('Edad', min_value=0, max_value=150, value=int(data_table.loc[edit_row_number, 'Edad (años)']))
                    Peso = st.number_input('Peso', min_value=0.0, value=float(data_table.loc[edit_row_number, 'Peso (kg)']))
                    Altura = st.number_input('Altura', min_value=0.0, value=float(data_table.loc[edit_row_number, 'Altura (cm)']))
                    Grasa = st.number_input('Grasa', min_value=0.0, value=float(data_table.loc[edit_row_number, 'Grasa (%)']))
                    Musculo = st.number_input('Musculo', min_value=5.0, value=float(data_table.loc[edit_row_number, 'Musculo (kg)']))
                    PBrazo = st.number_input('CMB', min_value=0.0, value=float(data_table.loc[edit_row_number, 'PBrazo (cm)']))
                    PPantorrilla = st.number_input('CMP', min_value=0.0, value=float(data_table.loc[edit_row_number, 'PPantorrilla (cm)']))
                    FA = st.number_input('FA', min_value=0.0, value=float(data_table.loc[edit_row_number, 'FA (kg)']))
                    Marcha = st.number_input('Marcha', min_value=0.0, value=float(data_table.loc[edit_row_number, 'Marcha (ms-1)']))

        
                    if st.form_submit_button('Guardar Cambios'):
                        # Actualiza la fila en data_table
                        data_table.loc[edit_row_number] = [Folio, Edad, Peso, Altura, Grasa, Musculo, PBrazo, PPantorrilla, FA, Marcha]
                        st.session_state.data = data_table
                        st.success('Cambios guardados con éxito!')

            # Mostrar los datos ingresados en el DataFrame
            if st.button('Mostrar Resultados'):
                st.subheader('DataFrame Resultante')
                st.write(st.session_state.data)

            # Botón para descargar los datos en formato Excel
            if not st.session_state.data.empty:
                st.subheader('Descargar Datos')
                st.write('Haga clic en el enlace a continuación para descargar los datos en formato Excel.')
    
                # Generar un enlace para la descarga del archivo Excel
                output = io.BytesIO()
                excel_writer = pd.ExcelWriter(output, engine='xlsxwriter')
                st.session_state.data.to_excel(excel_writer, sheet_name='Datos', index=False)
                excel_writer.save()
    
                # Crear el enlace de descarga
                excel_data = output.getvalue()
                b64 = base64.b64encode(excel_data).decode('utf-8')
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="datos_pacientes.xlsx">Descargar Excel</a>'
                st.markdown(href, unsafe_allow_html=True)




############







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
               "CV": "https://scholar.google.com.mx/citations?hl=es&user=TGZGewEAAAAJ", "contacto": "elena.bricio@colima.tecnm.mx"},
               {
               "nombre": "Dra. Mónica Ríos Silva",
               "foto": "rios.jpg",
               "reseña": "Médica cirujana y partera con especialidad en Medicina Interna y Doctorado en Ciencias Médicas por la Universidad de Colima, médica especialista del Hospital Materno Infantil de Colima y PTC de la Facultad de Medicina de la Universidad de Colima. Es profesora de los posgrados en Ciencias Médicas, Ciencias Fisiológicas, Nutrición clínica y Ciencia ambiental global.",
               "CV": "https://scholar.google.com.mx/scholar?hl=en&as_sdt=0%2C5&q=Monica+Rios+silva&btnG=", "contacto": "mrios@ucol.mx"},
               {
               "nombre": "Dra. Rosa Yolitzy Cárdenas María",  
               "foto": "cardenas.jpg",
               "reseña": "Ha realizado los estudios de Química Farmacéutica Bióloga, Maestría en Ciencias Médicas y Doctorado en Ciencias Médicas, todos otorgados por la Universidad de Colima. Actualmente, se desempeña como Técnica Académica Titular C en el Centro Universitario de Investigaciones Biomédicas de la Universidad de Colima, enfocándose en la investigación básica y clínica de enfermedades crónico-degenerativas no transmisibles en investigación. También es profesora en la Maestría y Doctorado en Ciencias Médicas, así como en la Maestría en Nutrición Clínica de la misma universidad. Es miembro del Sistema Nacional de Investigadores nivel I y miembro fundador activo de la asociación civil DAYIN (https://www.dayinac.org/)",
               "CV": "https://scholar.google.com.mx/scholar?hl=en&as_sdt=0%2C5&q=rosa+yolitzy+c%C3%A1rdenas-mar%C3%ADa&btnG=&oq=rosa+yoli", "contacto": "rosa_cardenas@ucol.mx"}
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
