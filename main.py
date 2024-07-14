
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
       <div style="text-align: justify;">
       
       Esta aplicación es resultado del proyecto de estancia posdoctoral "**Identificación 
       de las etapas y tipos de sarcopenia mediante modelos predictivos como herramienta 
       de apoyo en el diagnóstico a partir de parámetros antropométricos**", desarrollado 
       por el Doctor en Ciencias (Astrofísica) Santiago Arceo Díaz, bajo la dirección de 
       la Doctora Xóchitl Rosío Angélica Trujillo Trujillo, y con la ayuda de los colaboradores mencionados en esta sección. Esta estancia es gracias a la 
       colaboración entre el entre el **Consejo Nacional de Humanidades Ciencia y Tecnología ([**CONAHCYT**](https://conahcyt.mx/)) y la Universidad de Colima ([**UCOL**](https://portal.ucol.mx/cuib/))**
       </div>
       """, unsafe_allow_html=True)
       st.subheader("Muestra")
       
       st.markdown("""
       <div style="text-align: justify;">
       
       Los datos utilizados para los modelos se recolectaron a partir de un grupo de voluntarios de centros 
       de convivencia de personas adultas mayores, residentes en las Ciudades de Colima y Villa de Álvarez.
       **En la presente aplicación se crean modelos que permiten estimar variables como el porcentaje de grasa 
       corporal y masa muscular en personas adultas mayores**, permitiendo la evaluación de síndromes geriátricos 
       como la sarcopenia en situaciones en las que no se cuente con el equipo de medición adecuado.
       </div>
       """, unsafe_allow_html=True)

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
       <div style="text-align: justify;">

       En esta pestaña se muestra el proceso para calcular **modelos de aproximación** a la masa muscular, medida en kilogramos, y el porcentaje de grasa corporal **a partir de una sola variable**. Estos a su vez se usan en el módulo de ***"Estimación de riesgo de sarcopenia"*** (accesible en la barra lateral izquierda).
                   
       En la primera sección se muestran los modelos para la predicción de masa muscular y grasa mediante variables predefinidas. En el caso de la masa muscular, se predicen valores para pacientes  a partir del [**perímetro de pantorrilla**](https://scielo.isciii.es/pdf/nh/v33n3/10_original9.pdf) y en el caso de la grasa corporal se utiliza el [**perímetro de brazo**](https://www.sciencedirect.com/science/article/pii/S0212656709006416).
                   
       En la segunda sección se dejan las variables predictoras a elección del usuario y con ellas se crean modelos predictivos para la masa muscular total y el porcentaje de grasa corporal.
                   
       Los modelos predictivos para masa muscular y porcentaje de grasa corporal hechos a partir de diversos algoritmos basados en [**árboles de regresión**](https://www.researchgate.net/publication/242370834_Classification_and_Regression_Trees_An_Introduction). Dichos algoritmos son: *'árbol de regresión simple'*, *'Random forest'* y *'Gradient boosting'*. Si bien, la precisión de los modelos es limitada, presentan la ventaja de solo requerir variables atropométricas que pueden registrarse mediante una cinta métrica, permitiendo una estimación en casos en los que no se cuenta de otros intrumentos de medición. Así mismo se incluyeron modelos de ajuste lineal para establecer un punto de comparación con los modelos de árbol.
                   
       </div>                                           
       """, unsafe_allow_html=True)

       st.header("Estimación para masa muscular y porcentaje de grasa corporal usando los perímetros de pantorrilla y brazo.")

       st.markdown("""
       <div style="text-align: justify;">

       Los modelos que aquí se muestran fueron calculados a partir de una muestra con datos antropométricos de adultos mayores que asistem regularmente a centros de convivencia en la zona Colima-Villa de Álvarez y están limitados por el tamaño de la muestra. Así mismo existe un sezgo en el sexo de los participantes que es necesario tomar en cuenta, ya que **la mayoría de los participantes son mujeres**). Se espera que el efecto de este sesgo se reduzca cuando se recoleten mas datos.                              
       
        A continuación se muestra la base de datos de adultos mayores. La pestaña desplegable de "**Claves de variables**" explica que es cada una de estas variables. En la parte superior de cada columna se muestra el nombre del parámetro y las unidades correspondientes. Si deja el ícono del mouse en la parte superior derecha puede descargar la tabla con los datos.
       </div>
       """, unsafe_allow_html=True)

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
               st.write("Hay datos faltantes en algunas filas.")
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
       
       # Muestra una tabla con los primeros registros de los datos
       st.markdown("""
       <div style="text-align: justify;">
       Antes de crear los modelos, eliminamos las filas que tuvieran datos faltantes en alguna de las columnas de interés. La tabla con los datos que se usaron para los modelos se muestra a continuación:
                   
       </div>
       """, unsafe_allow_html=True) 
       
       data = df

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

       # Dividir la página en dos columnas
       col1, col2 = st.columns(2)
       # Agregar un botón de descarga para el dataframe en la primera columna
       with col1:
           download_button(data, 'muestra_antrompométrica_colima_2023_sin_nan.xlsx', 'Descargar como Excel')
           st.write('')
       # Agregar un botón de descarga para el dataframe en la segunda columna
       with col2:
           download_button_CSV(data, 'muestra_antrompométrica_colima_2023_sin_nan.csv', 'Descargar como CSV')
           st.write('')


       with st.expander("**¿Por qué predecir con mas de un modelo?**"):
           

        st.markdown("""
        <div style="text-align: justify;">           

        En el campo de la ciencia de datos y el aprendizaje automático, no existe un único modelo que sea el mejor para todos los problemas. Diferentes modelos tienen distintas fortalezas y debilidades, y su rendimiento puede variar significativamente dependiendo de la naturaleza de los datos con los que se entrenan. Al utilizar varios modelos, podemos obtener una mejor comprensión de los datos y evaluar cuál modelo proporciona las predicciones más precisas.

        ***¿Cómo elegir el mejor modelo?***
        Para determinar cuál modelo es el mejor, consideramos varias métricas de rendimiento. Las dos métricas clave utilizadas en este análisis son:

        ***Coeficiente de Determinación (R²):*** El R² mide la proporción de la variación en la variable dependiente (masa muscular) que es explicada por la variable independiente (perímetro de pantorrilla) en el modelo. Un valor de R² más alto indica que el modelo explica mejor la variabilidad de los datos.

        ***Error Absoluto Medio (MAE):*** El MAE mide la media de los errores absolutos entre las predicciones y los valores reales. Un MAE más bajo indica que las predicciones del modelo están, en promedio, más cerca de los valores reales.
        
        **Tipos de modelos**
        - **Regresión Lineal:** Es un modelo simple que asume una relación lineal entre las variables. Es fácil de interpretar, pero puede no capturar relaciones complejas en los datos.
        - **Árbol de Decisión:** Divide los datos en subconjuntos más pequeños basándose en características importantes, pero puede sobreajustarse a los datos de entrenamiento.
        - **Random Forest:** Combina múltiples árboles de decisión para mejorar la precisión y reducir el sobreajuste, pero puede ser más difícil de interpretar.
        - **Gradient Boosting:** Mejora iterativamente el modelo para corregir los errores de predicciones anteriores, ofreciendo alta precisión pero a costa de mayor complejidad y tiempo de entrenamiento.
        
                    
        **Elección del mejor modelo**
        Para elegir el mejor modelo, considera el contexto y las necesidades específicas de tu análisis:

        - **Precisión:** Si necesitas el modelo más preciso, selecciona el que tenga el R² más alto y el MAE más bajo.
        - **Interpretabilidad:** Si necesitas entender claramente cómo el modelo toma decisiones, puede que prefieras la regresión lineal o el árbol de decisión.
        - **Robustez:** Para un modelo equilibrado que maneje bien los datos nuevos y no vistos, Random Forest o Gradient Boosting suelen ser buenas opciones.

        </div>            
        """, unsafe_allow_html=True)


       import streamlit as st
       import pandas as pd
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor
       import numpy as np
       import matplotlib.pyplot as plt
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       from sklearn.metrics import mean_absolute_error


       st.set_option('deprecation.showPyplotGlobalUse', False)

       ########################
       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)']]
       y = data['Músculo (kg)']

       # Crea un título para la aplicación
       st.subheader('Predicción de la masa muscular')

       st.markdown("""
       <div style="text-align: justify;">
                   
       Como primera parte, se estimó la masa muscular de los participantes a partir del perímetro de pantorrilla. Se crearon modelos de ***regresión lineal***, ***Árbol de decisión***, ***Random Forest*** y ***Gradient Boosting***. La gráfica siguiente muestra las mediciones y las predicciones de los modelos.
                                       
       </div>
       """, unsafe_allow_html=True)
    

       # Crea un modelo de regresión lineal
       modelo_musculo_lr = LinearRegression()
       modelo_musculo_lr.fit(X, y)

       # Crea un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor()
       modelo_musculo_dt.fit(X, y)

       # Crea un modelo de Random Forest
       modelo_musculo_rf = RandomForestRegressor()
       modelo_musculo_rf.fit(X, y)

       # Crea un modelo de Gradient boosting
       modelo_musculo_gb = GradientBoostingRegressor()
       modelo_musculo_gb.fit(X, y)

       # Realiza predicciones para diferentes valores de PPantorrilla (cm)
       ppantorrilla_values = np.linspace(min(X['PPantorrilla (cm)']), max(X['PPantorrilla (cm)']), 100).reshape(-1, 1)
       musculo_pred_lr = modelo_musculo_lr.predict(ppantorrilla_values)
       musculo_pred_dt = modelo_musculo_dt.predict(ppantorrilla_values)
       musculo_pred_rf = modelo_musculo_rf.predict(ppantorrilla_values)
       musculo_pred_gb = modelo_musculo_gb.predict(ppantorrilla_values)
    
       # Calcula el coeficiente de determinación (R^2) para cada modelo
       r2_musculo_lr = modelo_musculo_lr.score(X, y)
       r2_musculo_dt = modelo_musculo_dt.score(X, y)
       r2_musculo_rf = modelo_musculo_rf.score(X, y)
       r2_musculo_gb = modelo_musculo_gb.score(X, y)

       #Calcula el error absoluto medio para cada modelo
       mae_lr = mean_absolute_error(y, modelo_musculo_lr.predict(X))
       mae_dt = mean_absolute_error(y, modelo_musculo_dt.predict(X))
       mae_rf = mean_absolute_error(y, modelo_musculo_rf.predict(X))
       mae_gb = mean_absolute_error(y, modelo_musculo_gb.predict(X))

       # Grafica los datos y las predicciones para cada modelo
      
       fig, ax=plt.subplots()
       ax.scatter(X, y, color = 'blue', label=f"Mediciones")       
       
       ax.plot(ppantorrilla_values, musculo_pred_lr, color='red', label=f'Regresión lineal (R^2={r2_musculo_lr:.2f})')
       ax.plot(ppantorrilla_values, musculo_pred_dt, color='green', label=f'Árbol de decisión (R^2={r2_musculo_dt:.2f})')
       ax.plot(ppantorrilla_values, musculo_pred_rf, color='blue', label=f'Random Forest (R^2={r2_musculo_rf:.2f})')
       ax.plot(ppantorrilla_values, musculo_pred_gb, color='purple', label=f'Gradient Boosting (R^2={r2_musculo_gb:.2f})')
    
       # Modificar el tamaño de fuente de las etiquetas de las líneas en el gráfico
       for label in ax.get_xticklabels() + ax.get_yticklabels():
           label.set_fontsize(8)

       ax.set_xlabel('Pantorrilla (cm)')
       ax.set_ylabel('Masa muscular (Kg)')
       ax.set_title('Predicciones para la masa muscular')
       ax.legend(fontsize='xx-small', loc='best')  # Modifica el tamaño de letra de las leyendas
       st.pyplot(fig)

       st.markdown("""
       <div style="text-align: justify;">
                              
       El **eje vertical** corresponde a la **masa muscular** y el **horizontal** al **perímetro de pantorrilla**. Las mediciones de masa muscular se registraron mediante una balanza de bioimpedancia y están representadas por los puntos azules. Las trayectorias de cada color muestran: el modelos de ***regresión lineal*** (representado por la línea roja), el modelo de ***árbol de decisión*** (linea verde), el modelo de ***Random Forest*** (linea azul) y el modelo de ***Gradient Boosting*** (linea púrpura). En la esquina inferior derecha se muestran los *coeficientes de determinación (R^2)* correspondientes a cada modelo. En la pestaña contraible ***'Coeficientes de ajuste para los modelos'*** muestran estos coeficientes y el *Error absoluto medio (MAE)*.
       </div>
       """, unsafe_allow_html=True) 

       # Coeficientes de ajuste para el modelo de regresión lineal       
       pendiente_musculo_lr = modelo_musculo_lr.coef_[0]
       intercepto_musculo_lr = modelo_musculo_lr.intercept_
       

       with st.expander("**Coeficientes de ajuste para los modelos**"):
    
           st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_musculo_lr}, **Intercepto** = {intercepto_musculo_lr}')
           st.write("***Coeficiente de determinación***")
           # Coeficientes de determinación (R^2) para los modelos
           st.write(f'**R^2 Regresión lineal:** {r2_musculo_lr}')       
           st.write(f'**R^2 Árbol de Decisión:** {r2_musculo_dt}')
           st.write(f'**R^2 Random Forest:** {r2_musculo_rf}')
           st.write(f'**R^2 Gradient Boosting:** {r2_musculo_gb}')
           st.write("***Error medio absoluto***")
           st.write(f'**MAE Regresión lineal:** {mae_lr:.2f}')
           st.write(f'**MAE Árbol de Decisión:** {mae_dt:.2f}')
           st.write(f'**MAE Random Forest:** {mae_rf:.2f}')
           st.write(f'**MAE Gradient Boosting:** {mae_gb:.2f}')


    

       import streamlit as st
       import matplotlib.pyplot as plt
       from sklearn.tree import DecisionTreeRegressor, plot_tree
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

       st.markdown("""
       <div style="text-align: justify;">
                   
       A continuación puede teclear un valor de perímetro de pantorrilla y al presionar *Enter* se muestra la estimación de masa muscular de acuerdo con cada modelo.
                   
       </div>
       """, unsafe_allow_html=True)

       # Input del usuario para el valor de la variable predictora
       input_value = st.number_input(f'**Introduzca un valor para el perímetro de pantorrilla**', min_value=float(X['PPantorrilla (cm)'].min()), max_value=float(X['PPantorrilla (cm)'].max()))

       # Realiza predicciones usando el valor de entrada del usuario
       input_array = np.array([[input_value]])
       prediction_lr = modelo_musculo_lr.predict(input_array)[0]
       prediction_dt = modelo_musculo_dt.predict(input_array)[0]
       prediction_rf = modelo_musculo_rf.predict(input_array)[0]
       prediction_gb = modelo_musculo_gb.predict(input_array)[0]

       # Muestra las predicciones
       st.write(f'**Predicción usando *Regresión Lineal*:** {prediction_lr:.2f} kg')
       st.write(f'**Predicción usando *Árbol de Decisión*:** {prediction_dt:.2f} kg')
       st.write(f'**Predicción usando *Random Forest*:** {prediction_rf:.2f} kg')
       st.write(f'**Predicción usando *Gradient Boosting*:** {prediction_gb:.2f} kg')
 


       # Crear un modelo de árbol de decisión limitando la profundidad
       #modelo_musculo_dt_simplified = DecisionTreeRegressor(max_depth=4)  # #Ajusta el valor de max_depth según sea necesario
       #modelo_musculo_dt_simplified.fit(X, y)

       # Generar el diagrama del árbol de decisión simplificado con un tamaño de letra más grande
       #figur = plt.figure(figsize=(40, 15), dpi=600)
       #plt.rc('font', size=12)  # Ajusta el tamaño de fuente aquí
       #plot_tree(modelo_musculo_dt_simplified, filled=True, feature_names=X.#columns, fontsize=12)  # Ajusta el tamaño de la letra aquí
       #plt.title("Árbol de Regresión para predecir la masa muscular (kg) a #partir del perímetro de pantorrilla (cm)", fontsize=30)  

       # Mostrar la figura en Streamlit
       #st.pyplot(figur)

       ####################

       import streamlit as st
       import pandas as pd
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       import numpy as np
       import matplotlib.pyplot as plt
       from sklearn.metrics import mean_absolute_error


       st.set_option('deprecation.showPyplotGlobalUse', False)
       st.subheader("Predicción del porcentaje de grasa corporal")
       st.markdown("""
       <div style="text-align: justify;">
       
       En esta sección se crean los modelos predictivos para el porcentaje de grasa corporal. Como variable predictora se seleccionó el ***perímetro de brazo***. 
                   
       La siguiente gráfica muestra las mediciones y las predicciones de los modelos. Los ejes vertical y horizontal corresponden al porcentaje de grasa corporal y al perímetro de brazo, respectivamente. Los puntos verdes corresponden a las mediciones tomadas de los participantes y las curvas de diferentes colores representan las estimaciones de cada modelo: el modelo de ***Regresión lineal*** (curva roja), el modelo de ***Árbol de decisión*** (curva verde), el de ***Random Forest*** (curva azul) y el de ***Gradient Boosting*** (curva púrpura). 
                
       </div>
       """, unsafe_allow_html=True)

       # Supongamos que 'df' es tu DataFrame ya cargado previamente
       data = df

       # Crear un título para la aplicación
       #st.subheader('Modelos de Regresión para Predicción de Músculo')

       # Mostrar una tabla con los primeros registros de los datos 
       #st.markdown("Esta es la base de datos con parámetros antropométricos:")
       #st.dataframe(data)

       #with st.expander("**Información adicional**"):
        ## Mostrar información adicional sobre el DataFrame
        #num_rows, num_columns = data.shape
        #missing_data = data.isnull().any().any()

        #st.write(f"**Número de filas**: {num_rows}")
        #st.write(f"**Número de columnas**: {num_columns}")
        #if missing_data:
        #    st.write("Existen datos faltantes en alguna fila.")
        #else:
        #    st.write("No hay datos faltantes en ninguna fila.")
    
####################

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor
       from sklearn.metrics import r2_score
       from sklearn.model_selection import RandomizedSearchCV
       from scipy.stats import uniform, randint
       X = data[['PBrazo (cm)']]
       y = data['Grasa Corporal (%)']

    
       # Crea un modelo de regresión lineal
       modelo_grasa_lr = LinearRegression()
       modelo_grasa_lr.fit(X, y)

       # Crea un modelo de árbol de decisión
       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X, y)

       # Crea un modelo de Random Forest
       modelo_grasa_rf = RandomForestRegressor()
       modelo_grasa_rf.fit(X, y)

       # Crear un modelo de Gradient Boosting para Grasa Corporal (%) vs. PBrazo (cm)
       modelo_grasa_gb = GradientBoostingRegressor()
       modelo_grasa_gb.fit(X, y)

       # Realiza predicciones para diferentes valores de PPantorrilla (cm)
       pbrazo_values = np.linspace(min(X['PBrazo (cm)']), max(X['PBrazo (cm)']), 100).reshape(-1, 1)
       grasa_pred_lr = modelo_grasa_lr.predict(pbrazo_values)
       grasa_pred_dt = modelo_grasa_dt.predict(pbrazo_values)
       grasa_pred_rf = modelo_grasa_rf.predict(pbrazo_values)
       grasa_pred_gb = modelo_grasa_gb.predict(pbrazo_values)

       # Calcula el coeficiente de determinación (R^2) para cada modelo
       r2_grasa_lr = modelo_grasa_lr.score(X, y)
       r2_grasa_dt = modelo_grasa_dt.score(X, y)
       r2_grasa_rf = modelo_grasa_rf.score(X, y)
       r2_grasa_gb = modelo_grasa_gb.score(X, y)

       from sklearn.metrics import mean_absolute_error

       # Cálculo de los errores absolutos medios (MAE) para cada modelo
       mae_lr = mean_absolute_error(y, modelo_grasa_lr.predict(X))
       mae_dt = mean_absolute_error(y, modelo_grasa_dt.predict(X))
       mae_rf = mean_absolute_error(y, modelo_grasa_rf.predict(X))
       mae_gb = mean_absolute_error(y, modelo_grasa_gb.predict(X))

       # Grafica los datos y las predicciones para cada modelo
       

       #st.write("En esta gráfica se comparan los modelos con los datos medidos (puntos azules). Las curvas de distintos colores correponden a: modelo lineal (en rojo), aproximación de Random Forest (azul) y aproximación de árbol de decisión (verde)")
       
       fig, ax=plt.subplots()
       ax.scatter(X, y, color = 'green', label="Mediciones")       
       ax.plot(pbrazo_values, grasa_pred_lr, color='red', label=f'Regresión lineal (R^2={r2_grasa_lr:.2f})')
       ax.plot(pbrazo_values, grasa_pred_dt, color='green', label=f'Árbol de decisión (R^2={r2_grasa_dt:.2f})')
       ax.plot(pbrazo_values, grasa_pred_rf, color='blue', label=f'Random forest (R^2={r2_grasa_rf:.2f})')
       ax.plot(pbrazo_values, grasa_pred_gb, color='purple', label=f'Gradient Boosting (R^2={r2_grasa_gb:.2f})')

       # Modificar el tamaño de fuente de las etiquetas de las líneas en el gráfico
       for label in ax.get_xticklabels() + ax.get_yticklabels():
           label.set_fontsize(8)

       ax.set_xlabel('Perímetro de brazo (cm)')
       ax.set_ylabel('Porcentaje de grasa corporal')
       ax.set_title('Predicciones para el porcentaje de grasa corporal')
       ax.legend(fontsize='xx-small', loc='best')  # Modifica el tamaño de letra de las leyendas
       st.pyplot(fig)

       with st.expander("**Coeficientes de ajuste para los modelos**"):
    
           st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_musculo_lr}, **Intercepto** = {intercepto_musculo_lr}')
           st.write("***Coeficiente de determinación***")
           # Coeficientes de determinación (R^2) para los modelos
           st.write(f'**R^2 Regresión lineal:** {r2_musculo_lr}')       
           st.write(f'**R^2 Árbol de Decisión:** {r2_musculo_dt}')
           st.write(f'**R^2 Random Forest:** {r2_musculo_rf}')
           st.write(f'**R^2 Gradient Boosting:** {r2_musculo_gb}')
           st.write("***Error medio absoluto***")
           st.write(f'**MAE Regresión lineal:** {mae_lr:.2f}')
           st.write(f'**MAE Árbol de Decisión:** {mae_dt:.2f}')
           st.write(f'**MAE Random Forest:** {mae_rf:.2f}')
           st.write(f'**MAE Gradient Boosting:** {mae_gb:.2f}')


       import streamlit as st
       import matplotlib.pyplot as plt
       from sklearn.tree import DecisionTreeRegressor, plot_tree
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

       st.markdown("""
       <div style="text-align: justify;">
                   
       A continuación puede teclear un valor de ***perímetro de brazo*** y al presionar *Enter* se muestra la estimación del ***porcentaje de grasa corporal*** de acuerdo con cada modelo.
                   
       </div>
       """, unsafe_allow_html=True)

       # Input del usuario para el valor de la variable predictora
       input_value = st.number_input(f'**Introduzca un valor para el perímetro de brazo**', min_value=float(X['PBrazo (cm)'].min()), max_value=float(X['PBrazo (cm)'].max()))

       # Realiza predicciones usando el valor de entrada del usuario
       input_array = np.array([[input_value]])
       prediction_lr = modelo_grasa_lr.predict(input_array)[0]
       prediction_dt = modelo_grasa_dt.predict(input_array)[0]
       prediction_rf = modelo_grasa_rf.predict(input_array)[0]
       prediction_gb = modelo_grasa_gb.predict(input_array)[0]

       # Muestra las predicciones
       st.write(f'**Predicción usando *Regresión Lineal*:** {prediction_lr:.2f} kg')
       st.write(f'**Predicción usando *Árbol de Decisión*:** {prediction_dt:.2f} kg')
       st.write(f'**Predicción usando *Random Forest*:** {prediction_rf:.2f} kg')
       st.write(f'**Predicción usando *Gradient Boosting*:** {prediction_gb:.2f} kg')

    
       st.subheader("Predicción de masa muscular y porcentaje de grasa corporal mediante otras variables antropométricas")

       st.markdown("""
       <div style="text-align: justify;">
                   
       En esta sección se deja a **elección del usuario** las variables que se utilizarán como predictores de ***masa muscular*** y ***porcentaje de grasa corporal***. La lista de variables posibles corresponde a las que ya de mostraron arriba (en la pestaña de ***Claves de variables***). Puede elegir las variables predictoras en cada uno de los menús desplegables. 
                   
       </div>
       """, unsafe_allow_html=True)

        # Obtener las variables numéricas del DataFrame
       numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
       numeric_columns.remove('Músculo (kg)')  # Remover la variable dependiente de las opciones

       # Selector de la variable predictora
       predictor = st.selectbox("***Selecciona la variable predictora para la masa muscular total (kg)***", numeric_columns)

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[[predictor]]
       y = data['Músculo (kg)']

       # Crea un modelo de regresión lineal
       modelo_musculo_lr = LinearRegression()
       modelo_musculo_lr.fit(X, y)

       # Crea un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor()
       modelo_musculo_dt.fit(X, y)

       # Crea un modelo de Random Forest
       modelo_musculo_rf = RandomForestRegressor()
       modelo_musculo_rf.fit(X, y)

       # Crea un modelo de Gradient Boosting
       modelo_musculo_gb = GradientBoostingRegressor()
       modelo_musculo_gb.fit(X, y)

       # Realiza predicciones para diferentes valores de la variable predictora seleccionada

       predictor_values = np.linspace(min(X[predictor]), max(X[predictor]), 100).reshape(-1, 1)
       musculo_pred_lr = modelo_musculo_lr.predict(predictor_values)
       musculo_pred_dt = modelo_musculo_dt.predict(predictor_values)
       musculo_pred_rf = modelo_musculo_rf.predict(predictor_values)
       musculo_pred_gb = modelo_musculo_gb.predict(predictor_values)

       # Calcula el coeficiente de determinación (R^2) para cada modelo
       r2_musculo_lr = modelo_musculo_lr.score(X, y)
       r2_musculo_dt = modelo_musculo_dt.score(X, y)
       r2_musculo_rf = modelo_musculo_rf.score(X, y)
       r2_musculo_gb = modelo_musculo_gb.score(X, y)


       #Calcula el error absoluto medio para cada
       # Cálculo de los errores absolutos medios (MAE) para cada modelo
       mae_lr = mean_absolute_error(y, modelo_musculo_lr.predict(X))
       mae_dt = mean_absolute_error(y, modelo_musculo_dt.predict(X))
       mae_rf = mean_absolute_error(y, modelo_musculo_rf.predict(X))
       mae_gb = mean_absolute_error(y, modelo_musculo_gb.predict(X))


       ####################

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor
       from sklearn.metrics import r2_score
       from sklearn.model_selection import RandomizedSearchCV
       from scipy.stats import uniform, randint

       import streamlit as st
       import pandas as pd
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       import numpy as np
       import matplotlib.pyplot as plt

       st.set_option('deprecation.showPyplotGlobalUse', False)

       # Supongamos que 'df' es tu DataFrame ya cargado previamente
       data = df

       # Obtener las variables numéricas del DataFrame
       numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
       numeric_columns.remove('Grasa Corporal (%)')  # Remover la variable dependiente de las opciones

       # Selector de la variable predictora
       predictor_2 = st.selectbox("***Seleccione la variable predictora para el porcentaje de grasa corporal***", numeric_columns)

       # Divide tus datos en características (X) y la variable dependiente (y)
       X_2 = data[[predictor_2]]
       y_2 = data['Grasa Corporal (%)']

       # Crea un modelo de regresión lineal
       modelo_grasa_lr = LinearRegression()
       modelo_grasa_lr.fit(X_2, y_2)

       # Crea un modelo de árbol de decisión
       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X_2, y_2)

       # Crea un modelo de Random Forest
       modelo_grasa_rf = RandomForestRegressor()
       modelo_grasa_rf.fit(X_2, y_2)

       # Crear un modelo de Gradient Boosting para Grasa Corporal (%) vs. PBrazo (cm)
       modelo_grasa_gb = GradientBoostingRegressor()
       modelo_grasa_gb.fit(X_2, y_2)

       # Realiza predicciones para diferentes valores de PPantorrilla (cm)
       #predictor_2_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
       #predictor_2_values = np.linspace(min(X_2[predictor_2]), max(X_2[predictor_2]), 100)
       predictor_2_values = np.linspace(min(X_2[predictor_2]), max(X_2[predictor_2]), 100).reshape(-1, 1) 
       #predictor_values = np.linspace(min(X[predictor]), max(X[predictor]), 100).reshape(-1, 1)
 
       grasa_pred_lr = modelo_grasa_lr.predict(predictor_2_values)
       grasa_pred_dt = modelo_grasa_dt.predict(predictor_2_values)
       grasa_pred_rf = modelo_grasa_rf.predict(predictor_2_values)
       grasa_pred_gb = modelo_grasa_gb.predict(predictor_2_values)

       # Calcula el coeficiente de determinación (R^2) para cada modelo
       r2_grasa_lr = modelo_grasa_lr.score(X_2, y_2)
       r2_grasa_dt = modelo_grasa_dt.score(X_2, y_2)
       r2_grasa_rf = modelo_grasa_rf.score(X_2, y_2)
       r2_grasa_gb = modelo_grasa_gb.score(X_2, y_2)

       from sklearn.metrics import mean_absolute_error

       # Cálculo de los errores absolutos medios (MAE) para cada modelo
       mae_lr = mean_absolute_error(y_2, modelo_grasa_lr.predict(X_2))
       mae_dt = mean_absolute_error(y_2, modelo_grasa_dt.predict(X_2))
       mae_rf = mean_absolute_error(y_2, modelo_grasa_rf.predict(X_2))
       mae_gb = mean_absolute_error(y_2, modelo_grasa_gb.predict(X_2))


       # Grafica los datos y las predicciones para cada modelo
       #st.write("**Gráfico de predicciones**")
       st.write("En esta gráfica se comparan los modelos con los datos medidos (puntos azules). Las curvas de distintos colores corresponden al modelo de ***Regresión lineal*** (en rojo), ***Random Forest*** (azul), ***Árbol de decisión*** (verde) y ***Gradient Boosting*** (morado).")
       fig, ax = plt.subplots()
       ax.scatter(X, y, color='blue', label="Mediciones")
       ax.plot(predictor_values, musculo_pred_lr, color='red', label=f'Regresión lineal (R^2={r2_musculo_lr:.2f})')
       ax.plot(predictor_values, musculo_pred_dt, color='green', label=f'Árbol de decisión (R^2={r2_musculo_dt:.2f})')
       ax.plot(predictor_values, musculo_pred_rf, color='blue', label=f'Random Forest (R^2={r2_musculo_rf:.2f})')
       ax.plot(predictor_values, musculo_pred_gb, color='purple', label=f'Gradient Boosting (R^2={r2_musculo_gb:.2f})')

       # Modificar el tamaño de fuente de las etiquetas de las líneas en el gráfico
       for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)

       ax.set_xlabel(predictor)
       ax.set_ylabel('Masa muscular (kg)')
       ax.set_title('Predicciones para la Masa muscular (kg)')
       ax.legend(fontsize='xx-small', loc='best')  # Modifica el tamaño de letra de las leyendas
       st.pyplot(fig)

       # Coeficientes de ajuste para el modelo de regresión lineal
       pendiente_musculo_lr = modelo_musculo_lr.coef_[0]
       intercepto_musculo_lr = modelo_musculo_lr.intercept_

       with st.expander("**Coeficientes de ajuste para los modelos**"):
    
           st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_musculo_lr}, **Intercepto** = {intercepto_musculo_lr}')
           st.write("***Coeficiente de determinación***")
           # Coeficientes de determinación (R^2) para los modelos
           st.write(f'**R^2 Regresión lineal:** {r2_musculo_lr}')       
           st.write(f'**R^2 Árbol de Decisión:** {r2_musculo_dt}')
           st.write(f'**R^2 Random Forest:** {r2_musculo_rf}')
           st.write(f'**R^2 Gradient Boosting:** {r2_musculo_gb}')
           st.write("***Error medio absoluto***")
           st.write(f'**MAE Regresión lineal:** {mae_lr:.2f}')
           st.write(f'**MAE Árbol de Decisión:** {mae_dt:.2f}')
           st.write(f'**MAE Random Forest:** {mae_rf:.2f}')
           st.write(f'**MAE Gradient Boosting:** {mae_gb:.2f}')


       fig, ax = plt.subplots()
       ax.scatter(X_2, y_2, color='green', label="Mediciones")
       ax.plot(predictor_2_values, grasa_pred_lr, color='red', label=f'Regresión lineal (R^2={r2_grasa_lr:.2f})')
       ax.plot(predictor_2_values, grasa_pred_dt, color='green', label=f'Árbol de decisión (R^2={r2_grasa_dt:.2f})')
       ax.plot(predictor_2_values, grasa_pred_rf, color='blue', label=f'Random Forest (R^2={r2_grasa_rf:.2f})')
       ax.plot(predictor_2_values, grasa_pred_gb, color='purple', label=f'Gradient Boosting (R^2={r2_grasa_gb:.2f})')

       # Modificar el tamaño de fuente de las etiquetas de las líneas en el gráfico
       for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)

       ax.set_xlabel(predictor_2)
       ax.set_ylabel('Grasa Corporal (%)')
       ax.set_title('Predicciones para el porcentaje de grasa corporal')
       ax.legend(fontsize='xx-small', loc='best')  # Modifica el tamaño de letra de las leyendas
       st.pyplot(fig)

       # Coeficientes de ajuste para el modelo de regresión lineal
       pendiente_grasa_lr = modelo_grasa_lr.coef_[0]
       intercepto_grasa_lr = modelo_grasa_lr.intercept_

       with st.expander("**Coeficientes de ajuste para los modelos**"):
    
           st.write(f'**Ajuste Lineal: Pendiente =** {pendiente_grasa_lr}, **Intercepto** = {intercepto_grasa_lr}')
           st.write("***Coeficiente de determinación***")
           # Coeficientes de determinación (R^2) para los modelos
           st.write(f'**R^2 Regresión lineal:** {r2_grasa_lr}')       
           st.write(f'**R^2 Árbol de Decisión:** {r2_grasa_dt}')
           st.write(f'**R^2 Random Forest:** {r2_grasa_rf}')
           st.write(f'**R^2 Gradient Boosting:** {r2_grasa_gb}')
           st.write("***Error medio absoluto***")
           st.write(f'**MAE Regresión lineal:** {mae_lr:.2f}')
           st.write(f'**MAE Árbol de Decisión:** {mae_dt:.2f}')
           st.write(f'**MAE Random Forest:** {mae_rf:.2f}')
           st.write(f'**MAE Gradient Boosting:** {mae_gb:.2f}')

       import streamlit as st
       import matplotlib.pyplot as plt
       from sklearn.tree import DecisionTreeRegressor, plot_tree
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

       st.markdown("""
       <div style="text-align: justify;">
                   
       A continuación puede teclear un valor de ***perímetro de brazo*** y al presionar *Enter* se muestra la estimación del ***porcentaje de grasa corporal*** de acuerdo con cada modelo.
                   
       </div>
       """, unsafe_allow_html=True)

       # Input del usuario para el valor de la variable predictora
       input_value = st.number_input(f'**Introduzca un valor para la variable predictora de masa muscular**', min_value=float(X[predictor].min()), max_value=float(X[predictor].max()))

       # Realiza predicciones usando el valor de entrada del usuario
       input_array = np.array([[input_value]])
       prediction_lr = modelo_musculo_lr.predict(input_array)[0]
       prediction_dt = modelo_musculo_dt.predict(input_array)[0]
       prediction_rf = modelo_musculo_rf.predict(input_array)[0]
       prediction_gb = modelo_musculo_gb.predict(input_array)[0]

       # Muestra las predicciones
       st.write(f'**Predicción usando *Regresión Lineal*:** {prediction_lr:.2f} kg')
       st.write(f'**Predicción usando *Árbol de Decisión*:** {prediction_dt:.2f} kg')
       st.write(f'**Predicción usando *Random Forest*:** {prediction_rf:.2f} kg')
       st.write(f'**Predicción usando *Gradient Boosting*:** {prediction_gb:.2f} kg')

      # Input del usuario para el valor de la variable predictora
       input_value_2 = st.number_input(f'**Introduzca un valor para la variable predictora del porcentaje de grasa corporal**', min_value=float(X_2[predictor_2].min()), max_value=float(X_2[predictor_2].max()))

       # Realiza predicciones usando el valor de entrada del usuario
       input_array = np.array([[input_value_2]])
       prediction_lr = modelo_grasa_lr.predict(input_array)[0]
       prediction_dt = modelo_grasa_dt.predict(input_array)[0]
       prediction_rf = modelo_grasa_rf.predict(input_array)[0]
       prediction_gb = modelo_grasa_gb.predict(input_array)[0]

       # Muestra las predicciones
       st.write(f'**Predicción usando *Regresión Lineal*:** {prediction_lr:.2f} %')
       st.write(f'**Predicción usando *Árbol de Decisión*:** {prediction_dt:.2f} %')
       st.write(f'**Predicción usando *Random Forest*:** {prediction_rf:.2f} %')
       st.write(f'**Predicción usando *Gradient Boosting*:** {prediction_gb:.2f} %')


       #####################

       #import streamlit as st
       #import matplotlib.pyplot as plt
       #from sklearn.tree import DecisionTreeRegressor, plot_tree

       # Crear un modelo de árbol de decisión limitando la profundidad
       #modelo_grasa_dt_simplified = DecisionTreeRegressor(max_depth=4)  # Ajusta el valor de max_depth según sea necesario
       #modelo_grasa_dt_simplified.fit(X, y)

       # Generar el diagrama del árbol de decisión simplificado con un tamaño de letra más grande
       #figur = plt.figure(figsize=(40, 15), dpi=600)
       #plt.rc('font', size=12)  # Ajusta el tamaño de fuente aquí
       #plot_tree(modelo_musculo_dt_simplified, filled=True, feature_names=X.#columns, fontsize=12)  # Ajusta el tamaño de la letra aquí
       #plt.title("Árbol de Regresión para predecir el porcentaje de grasa #corporal a partir del perímetro de brazo (cm)", fontsize=30)  # #Ajusta el tamaño de fuente del título aquí

       # Mostrar la figura en Streamlit
       #st.pyplot(figur)

       ##################################


# Contenido de la pestaña 2

       ##################################
# Contenido de la pestaña 2
elif pestañas == "Modelos con 2 variables":
       st.title("Modelos de aproximación con dos variables independientes")
       st.markdown("""
       <div style="text-align: justify;">
                   
       En esta sección de crean modelos predictivos para la ***masa muscular*** y el ***porcentaje de grasa corporal*** utilizando parejas de variables antropométricas. Si bien la complejidas de los modelos se incrementa, también lo hace su precisión. 
                                
      Los modelos predictivos para masa muscular y porcentaje de grasa corporal hechos a partir de diversos algoritmos basados en árboles de regresión. Dichos algoritmos son: ***'árbol de regresión simple'***, ***'Random forest'*** y ***'Gradient boosting'***. Si bien, la precisión de los modelos es limitada, presentan la ventaja de solo requerir variables atropométricas que pueden registrarse mediante una cinta métrica, permitiendo una estimación en casos en los que no se cuenta de otros intrumentos de medición. Así mismo se incluyeron modelos de ajuste lineal para establecer un punto de comparación con los modelos de árbol.
                   
       En la primera sección se muestran los modelos que utilizan el ***perímetro de pantorrilla y la fuerza de agarre*** para predecir la ***masa muscular*** y ***el perímetro de brazo y el pliegue cutáneo subescapular*** para predecir el ***porcentaje de grasa corporal.***
                   
       En la segunda sección se deja a elección del usuario las variables predictoras.
                   
       </div>
       """, unsafe_allow_html=True)
       
       ##import streamlit as st       
       #df=pd.read_excel('AM_2023_Antropo.xlsx')
       #st.dataframe(df)
       #df=df.dropna()
       #df['FA'] = (df['Fuerza mano derecha'] + df['Fuerza mano #izquierda']) / 2
       #df['Gs Brazo'] = (df['Gs Brazo derecho'] + df['Gs Brazo #izquierdo']) / 2
       #df['Gs Pierna'] = (df['Gs pierna derecha'] + df['Gs pierna #izquierda']) / 2
       #df=df[['Folio', 'Peso (kg)', 'Talla (cm)', 'IMC', 'PCintura (cm)',
       #       'PCadera (cm)', 'PBrazo (cm)', 'PPantorrilla (cm)', 'PCB (mm)#',
       #       'PCT (mm)', 'PCSE (mm)', 'Agua Corporal (%)', 'Músculo (kg)',
       #       'Grasa Corporal (%)', 'Centro',
       #       'FA','Velocidad de marcha']]
       
       st.markdown("""
       <div style="text-align: justify;">

       Los modelos que aquí se muestran fueron calculados a partir de una muestra con datos antropométricos de adultos mayores que asistem regularmente a centros de convivencia en la zona Colima-Villa de Álvarez y están limitados por el tamaño de la muestra. Así mismo existe un sezgo en el sexo de los participantes que es necesario tomar en cuenta, ya que **la mayoría de los participantes son mujeres**). Se espera que el efecto de este sesgo se reduzca cuando se recoleten mas datos.                              
       
        A continuación se muestra la base de datos de adultos mayores. La pestaña desplegable de "**Claves de variables**" explica que es cada una de estas variables. En la parte superior de cada columna se muestra el nombre del parámetro y las unidades correspondientes. Si deja el ícono del mouse en la parte superior derecha puede descargar la tabla con los datos.
       </div>
       """, unsafe_allow_html=True)

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
               st.write("Hay datos faltantes en algunas filas.")
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
       
       # Muestra una tabla con los primeros registros de los datos
       st.markdown("""
       <div style="text-align: justify;">
       Antes de crear los modelos, eliminamos las filas que tuvieran datos faltantes en alguna de las columnas de interés. La tabla con los datos que se usaron para los modelos se muestra a continuación:
                   
       </div>
       """, unsafe_allow_html=True) 
       
       data = df

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

       # Dividir la página en dos columnas
       col1, col2 = st.columns(2)
       # Agregar un botón de descarga para el dataframe en la primera columna
       with col1:
           download_button(data, 'muestra_antrompométrica_colima_2023_sin_nan.xlsx', 'Descargar como Excel')
           st.write('')
       # Agregar un botón de descarga para el dataframe en la segunda columna
       with col2:
           download_button_CSV(data, 'muestra_antrompométrica_colima_2023_sin_nan.csv', 'Descargar como CSV')
           st.write('')

       st.subheader("Modelos predictivos que usan fuerza de agarre y perímetro de pantorrilla.")
       
       import streamlit as st
       import pandas as pd
       from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
       from sklearn.ensemble import RandomForestRegressor
       import matplotlib.pyplot as plt
       
       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       import pandas as pd


       st.markdown("""
       <div style="text-align: justify;">
                   
       A continuación se muestran los algoritmos para el cálculo de la ***masa muscular*** en función de la ***fuerza de agarre*** y el ***perímetro de pantorrilla***. **Puede dar click en cada una de las pestañas de abajo para vizualizar una explicación de como cada algoritmo predice los datos contenidos en la muestra.** 
                   
       </div>
       """, unsafe_allow_html=True)

       # Supongamos que 'df' es tu DataFrame ya cargado previamente 
       data = df

       # Divide tus datos en características (X) y la variable dependiente (y)
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Crea un modelo de árbol de decisión
       modelo_musculo_dt = DecisionTreeRegressor(max_depth=4)
       modelo_musculo_dt.fit(X, y)

       # Extraer las reglas de decisión
       tree_rules = export_text(modelo_musculo_dt, feature_names=list(X.columns))

       # Crea un modelo de Random Forest
       modelo_musculo_rf = RandomForestRegressor(n_estimators=10)
       modelo_musculo_rf.fit(data[['PPantorrilla (cm)']], data['Músculo (kg)'])
       # Crea un modelo de Gradient Boosting
       modelo_musculo_gb = GradientBoostingRegressor(n_estimators=10)
       modelo_musculo_gb.fit(data[['PPantorrilla (cm)']], data['Músculo (kg)'])


       # Crear pestañas
       #tab1, tab2 = st.tabs(["Árbol de Decisión Simple", "Random Forest"])
       tab1, tab2, tab3, tab4 = st.tabs(["Regresión lineal", "Árbol de Decisión Simple", "Random Forest", "Gradient Boosting"])
       with tab1:
        import streamlit as st
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        import matplotlib.pyplot as plt
        import numpy as np
        st.markdown("""
        ### ¿Qué es la Regresión Lineal?
        La regresión lineal es un método estadístico utilizado para modelar la relación entre una variable dependiente ($Y$) y una o más variables independientes ($X$). En este modelo, asumimos que hay una relación lineal entre las variables.

        La fórmula de un modelo de regresión lineal simple es:
        """)

        st.latex(r'''
        Y = \beta_0 + \beta_1 X + \epsilon
        ''')

        st.markdown("""
        donde:
        - $Y$ es la variable dependiente (la que queremos predecir).
        - $\beta_0$ es la intersección o término constante.
        - $\beta_1$ es el coeficiente de la variable independiente.
        - $X$ es la variable independiente.
        - $\epsilon$ es el término de error o residual.

        Para múltiples variables independientes, el modelo se extiende a:
        """)

        st.latex(r'''
        Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_n X_n + \epsilon
        ''')

#        st.markdown("""
#            ### Aplicación a nuestro Conjunto de Datos
#            En este ejemplo, utilizamos el perímetro de la pantorrilla (PPantorrilla) y la Fuerza de Agarre (FA) como variables independientes para predecir la masa muscular (Músculo).

            #### Paso 1: Ajuste del Modelo
#            Primero, ajustamos un modelo de regresión lineal a nuestros datos.
#            """)


#        with st.expander("coeficientes del modelo"):
#            # Divide tus datos en características (X) y la variable dependiente (y)
#            X = data[['PPantorrilla (cm)', 'FA']]
#            y = data['Músculo (kg)']

#            # Crear el modelo de regresión lineal
#            modelo_musculo_lr = LinearRegression()
#            modelo_musculo_lr.fit(X, y)

#            # Realizar predicciones
#            y_pred = modelo_musculo_lr.predict(X)

#            # Calcular el coeficiente de determinación (R^2)
#            r2 = r2_score(y, y_pred)
#            mse = mean_squared_error(y, y_pred)
#            rmse = np.sqrt(mse)

#            # Mostrar los coeficientes del modelo
#            st.write("### Coeficientes del Modelo de Regresión Lineal:")
#            st.write(f"- Intercepto (\\(\\beta_0\\)): {modelo_musculo_lr.intercept_}")
#            st.write(f"- Coeficiente para PPantorrilla (\\(\\beta_1\\)): {modelo_musculo_lr.coef_[0]}")
#            st.write(f"- Coeficiente para FA (\\(\\beta_2\\)): {modelo_musculo_lr.coef_[1]}")

#            # Mostrar las métricas de evaluación del modelo
#            st.write("### Métricas de Evaluación del Modelo:")
#            st.write(f"- Coeficiente de Determinación (R²): {r2:.2f}")
#            st.write(f"- Error Cuadrático Medio (MSE): {mse:.2f}")
#            st.write(f"- Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")

        # Gráfica de las predicciones vs los valores reales
        #st.write("### Gráfica de las Predicciones vs los Valores Reales")
        #fig, ax = plt.subplots()
        #ax.scatter(y, y_pred, edgecolors=(0, 0, 0))
        #ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
        #ax.set_xlabel('Valores Reales')
        #ax.set_ylabel('Predicciones')
        #ax.set_title('Comparación de Valores Reales vs Predicciones')
        #st.pyplot(fig)

        #st.write("""
        ### Predicción de Masa Muscular
        #Utiliza los controles a continuación para ingresar los valores de PPantorrilla (cm) y FA, y el modelo de regresión lineal te proporcionará una predicción de la masa muscular.
        #""")

        # Interfaz de usuario para predicciones
        #ppantorrilla_input = st.number_input("Ingresa el valor de PPantorrilla (cm): ", min_value=20.0, max_value=50.0, value=30.0, step=0.1)
        #fa_input = st.number_input("Ingresa el valor de FA: ", min_value=30.0, max_value=80.0, value=55.0, step=0.1)

        #if st.button("Predecir Músculo"):
        #    prediccion = modelo_musculo_lr.predict([[ppantorrilla_input, fa_input]])[0]
        #    st.write(f"### Predicción de Músculo (kg): {prediccion:.2f}")



       with tab2:
        st.markdown("""

        Un árbol de decisión es un modelo predictivo que utiliza un conjunto de reglas basadas en las características de los datos para hacer predicciones. En un árbol de decisión, los datos se dividen en subconjuntos más pequeños basados en una característica que proporciona la mayor ganancia de información. Cada nodo del árbol representa una característica de los datos, cada rama representa un resultado de la característica y cada hoja representa una predicción.

        - Variables predictoras: PPantorrilla y FA.
        - Variable a predecir: Masa muscular (Músculo).
        El árbol de decisión sigue un camino desde la raíz hasta una hoja, aplicando reglas en cada nodo y decidiendo la predicción final basada en la hoja en la que termina.

        """)
        
        st.markdown("""
        ### Árbol de Decisión Simple
        El siguiente diagrama muestra el árbol de decisión simple que se utilizó para predecir la masa muscular basada en el perímetro de la pantorrilla (PPantorrilla) y la Fuerza de Agarre (FA).
        """)

        # Genera el diagrama del árbol de decisión
        fig = plt.figure(figsize=(20, 10))
        plot_tree(modelo_musculo_dt, filled=True, feature_names=X.columns, fontsize=10)
        plt.title("Árbol de Decisión para Músculo (kg) vs. PPantorrilla (cm) y FA", fontsize=14)
        st.pyplot(fig)


       with tab3: 
        
        st.markdown("""
        El Random Forest es un modelo de conjunto que construye múltiples árboles de decisión y los combina para mejorar la precisión y evitar el sobreajuste. Cada árbol en el bosque es entrenado con un subconjunto diferente de los datos (mediante bootstrap) y considera un subconjunto aleatorio de características para dividir en cada nodo. La predicción final es la media de las predicciones de todos los árboles individuales.

        - Variables predictoras: PPantorrilla.
        - Variable a predecir: Masa muscular (Músculo).
        El modelo Random Forest agrega la robustez de múltiples árboles de decisión, haciendo que sea menos susceptible al sobreajuste y proporcionando predicciones más precisas en general.

        """)
        
        
        st.markdown("""
        ### Visualización de Random Forest
        A continuación, se muestra uno de los árboles del modelo Random Forest. Puedes seleccionar el índice del árbol que deseas visualizar.
        """)

        # Visualización de uno de los árboles en el Random Forest
        if modelo_musculo_rf.n_estimators > 0:
            tree_index = st.number_input("Selecciona el índice del árbol a visualizar (0 a n_estimators-1):", min_value=0, max_value=modelo_musculo_rf.n_estimators-1, value=0, step=1)
            tree = modelo_musculo_rf.estimators_[tree_index]

            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(tree, feature_names=['PPantorrilla (cm)'], filled=True, ax=ax, fontsize=10)
            st.pyplot(fig)
        else:
            st.write("No hay árboles en el Random Forest.")

       with tab4:
        
        st.markdown("""

        El Gradient Boosting es otro modelo de conjunto que construye árboles de decisión de manera secuencial, cada uno intentando corregir los errores del árbol anterior. Cada árbol se ajusta a los residuos del árbol anterior en lugar de los valores originales. La predicción final es la suma de las predicciones de todos los árboles ajustados.

        - Variables predictoras: PPantorrilla.
        - Variable a predecir: Masa muscular (Músculo).
        El Gradient Boosting es potente y a menudo proporciona una alta precisión, pero puede ser más propenso al sobreajuste si no se regula correctamente.

        """)
        
        st.markdown("""
            ### Visualización de Gradient Boosting
            A continuación, se muestra uno de los árboles del modelo Gradient Boosting. Puedes seleccionar el índice del árbol que deseas visualizar.
            """)

        # Visualización de uno de los árboles en el Gradient Boosting
        if modelo_musculo_gb.n_estimators > 0:
         tree_index = st.number_input("Seleccionar el índice del modelo árbol a visualizar (0 a n_estimators-1):", min_value=0, max_value=modelo_musculo_gb.n_estimators-1, value=0, step=1)
         tree = modelo_musculo_gb.estimators_[tree_index][0]  # Cada estimador es una lista de un solo árbol

         fig, ax = plt.subplots(figsize=(20, 10))
         plot_tree(tree, feature_names=['PPantorrilla (cm)'], filled=True, ax=ax, fontsize=10)
         st.pyplot(fig)
        else:
         st.write("No hay árboles en el Gradient Boosting.")




       st.markdown("""
       <div style="text-align: justify;">
       
       Abajo puede ver las predicciones que cada modelo hace sobre el conjunto de datos. Los **gráficos de superficie** muestran los puntos que corresponden a las mediciones de ***fuerza de agarre*** y al ***perímetro de pantorrilla*** (representados por los puntos de distintos colores). El color de cada punto corresponde al rango de ***masa muscular*** al que corresponde dicho punto. En cuanto a los modelos, las zonas de cada color corresponden a los rangos que predicen. Se considera que el modelo es acertado en la predicción de masa muscular siempre que el color de la zona coincide con las de los puntos que están en ella. **Arriba de cada gráfico de superficie se muestra el coeficiente de determinación del modelo.**
       
       </div>
       """, unsafe_allow_html=True)

       
       
       data = df


       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       import pandas as pd

       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Define rangos de valores para PPantorrilla (cm) y FA
       ppantorilla_range = np.linspace(X['PPantorrilla (cm)'].min(), X['PPantorrilla (cm)'].max(), 100)
       fa_range = np.linspace(X['FA'].min(), X['FA'].max(), 100)
       ppantorilla_grid, fa_grid = np.meshgrid(ppantorilla_range, fa_range)

       # Combina las características en una matriz bidimensional
       X_grid = np.c_[ppantorilla_grid.ravel(), fa_grid.ravel()]

       # Lista de modelos y sus nombres
       modelos = {
            'Regresión Lineal': LinearRegression(),
            'Árbol de Decisión': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

       # Entrenar y predecir con cada modelo
       predicciones = {}
       r2_scores = {}

       for nombre, modelo in modelos.items():
            modelo.fit(X, y)
            y_pred = modelo.predict(X_grid)
            y_pred = y_pred.reshape(ppantorilla_grid.shape)
            predicciones[nombre] = y_pred
            r2_scores[nombre] = modelo.score(X, y)

       # Crear pestañas en Streamlit
       tabs = st.tabs(list(modelos.keys()))

       for i, nombre in enumerate(modelos.keys()):
            with tabs[i]:
                st.header(f"{nombre}")
                y_pred = predicciones[nombre]
                fig = plt.figure(figsize=(10, 6))
                contour = plt.contourf(ppantorilla_grid, fa_grid, y_pred, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
                plt.scatter(X['PPantorrilla (cm)'], X['FA'], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
                plt.xlabel('PPantorrilla (cm)')
                plt.ylabel('FA')
                plt.title(f'Superficie de Decisión de {nombre} para Músculo (kg) (R^2={r2_scores[nombre]:.2f})')
                cbar = plt.colorbar(contour)
                cbar.set_label('Músculo (kg)')
                st.pyplot(fig)


       # Divide tus datos en características (X) y la variable dependiente (y) 
       X = data[['PPantorrilla (cm)', 'FA']]
       y = data['Músculo (kg)']

       # Crear y ajustar los modelos 
       modelo_musculo_lr = LinearRegression()
       modelo_musculo_lr.fit(X, y)

       modelo_musculo_dt = DecisionTreeRegressor()
       modelo_musculo_dt.fit(X, y)

       modelo_musculo_rf = RandomForestRegressor()
       modelo_musculo_rf.fit(X, y)

       modelo_musculo_gb = GradientBoostingRegressor()
       modelo_musculo_gb.fit(X, y)

       # Realizar predicciones
       y_pred_lr = modelo_musculo_lr.predict(X)
       y_pred_dt = modelo_musculo_dt.predict(X)
       y_pred_rf = modelo_musculo_rf.predict(X)
       y_pred_gb = modelo_musculo_gb.predict(X)

       # Calcular el coeficiente de determinación (R^2) y RMSE para cada modelo
       metrics = {
        "Linear Regression": {
        "r2": r2_score(y, y_pred_lr),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_lr))
            },
        "Decision Tree": {
        "r2": r2_score(y, y_pred_dt),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_dt))
            },
        "Random Forest": {
        "r2": r2_score(y, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_rf))
            },
        "Gradient Boosting": {
        "r2": r2_score(y, y_pred_gb),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_gb))
            }
        }

       # Mostrar las métricas de evaluación del modelo
       with st.expander("**Métricas de Evaluación del Modelo:**"):
        #st.write("### Métricas de Evaluación del Modelo:")
        for model, values in metrics.items():
            st.write(f"**{model}**")
            st.write(f"- Coeficiente de Determinación (R²): {values['r2']:.2f}")
            st.write(f"- Raíz del Error Cuadrático Medio (RMSE): {values['rmse']:.2f}")

       # Interfaz de usuario para predicciones
       st.write("### Predicción de Masa Muscular")
       ppantorrilla_input = st.number_input("Ingresa el valor de PPantorrilla (cm): ", min_value=20.0, max_value=50.0, value=30.0, step=0.1)
       fa_input = st.number_input("Ingresa el valor de FA: ", min_value=30.0, max_value=80.0, value=55.0, step=0.1)

       if st.button("Predecir Músculo"):
            prediccion_lr = modelo_musculo_lr.predict([[ppantorrilla_input, fa_input]])[0]
            prediccion_dt = modelo_musculo_dt.predict([[ppantorrilla_input, fa_input]])[0]
            prediccion_rf = modelo_musculo_rf.predict([[ppantorrilla_input, fa_input]])[0]
            prediccion_gb = modelo_musculo_gb.predict([[ppantorrilla_input, fa_input]])[0]

            st.write(f"### Predicciones de Músculo (kg):")
            st.write(f"- **Regresión Lineal**: {prediccion_lr:.2f} kg")
            st.write(f"- **Árbol de Decisión**: {prediccion_dt:.2f} kg")
            st.write(f"- **Random Forest**: {prediccion_rf:.2f} kg")
            st.write(f"- **Gradient Boosting**: {prediccion_gb:.2f} kg")



       ######################       
       ######################

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.linear_model import LinearRegression 
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       from sklearn.metrics import mean_squared_error, r2_score 
       import pandas as pd

       # Supongamos que 'df' es tu DataFrame ya cargado previamente
       data = df

       X = data[['PBrazo (cm)', 'PCB (mm)']]
       y = data['Grasa Corporal (%)']

       # Define rangos de valores para PBrazo (cm) y PCB (mm)
       pbrazo_range = np.linspace(X['PBrazo (cm)'].min(), X['PBrazo (cm)'].max(), 100)
       pcb_range = np.linspace(X['PCB (mm)'].min(), X['PCB (mm)'].max(), 100)
       pbrazo_grid, pcb_grid = np.meshgrid(pbrazo_range, pcb_range)

       # Combina las características en una matriz bidimensional
       X_grid = np.c_[pbrazo_grid.ravel(), pcb_grid.ravel()]

       # Lista de modelos y sus nombres
       modelos = {
        'Regresión Lineal': LinearRegression(),
        'Árbol de Decisión': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
        }

       # Entrenar y predecir con cada modelo
       predicciones = {}
       r2_scores = {}

       for nombre, modelo in modelos.items():
            modelo.fit(X, y)
            y_pred = modelo.predict(X_grid)
            y_pred = y_pred.reshape(pbrazo_grid.shape)
            predicciones[nombre] = y_pred
            r2_scores[nombre] = modelo.score(X, y)

       # Crear pestañas en Streamlit
       tabs = st.tabs(list(modelos.keys()))

       for i, nombre in enumerate(modelos.keys()):
        with tabs[i]:
            st.header(f"{nombre}")
            y_pred = predicciones[nombre]
            fig = plt.figure(figsize=(10, 6))
            contour = plt.contourf(pbrazo_grid, pcb_grid, y_pred, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
            plt.scatter(X['PBrazo (cm)'], X['PCB (mm)'], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
            plt.xlabel('PBrazo (cm)')
            plt.ylabel('PCB (mm)')
            plt.title(f'Superficie de Decisión de {nombre} para Grasa Corporal (%) (R^2={r2_scores[nombre]:.2f})')
            cbar = plt.colorbar(contour)
            cbar.set_label('Grasa Corporal (%)')
            st.pyplot(fig)

       # Crear y ajustar los modelos 
       modelo_grasa_lr = LinearRegression() 
       modelo_grasa_lr.fit(X, y)

       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X, y)

       modelo_grasa_rf = RandomForestRegressor()
       modelo_grasa_rf.fit(X, y)

       modelo_grasa_gb = GradientBoostingRegressor() 
       modelo_grasa_gb.fit(X, y)

       # Realizar predicciones
       y_pred_lr = modelo_grasa_lr.predict(X)
       y_pred_dt = modelo_grasa_dt.predict(X)
       y_pred_rf = modelo_grasa_rf.predict(X)
       y_pred_gb = modelo_grasa_gb.predict(X)

       # Calcular el coeficiente de determinación (R^2) y RMSE para cada modelo
       metrics = {
            "Linear Regression": {
            "r2": r2_score(y, y_pred_lr),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_lr))
            },
            "Decision Tree": {
            "r2": r2_score(y, y_pred_dt),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_dt))
            },
            "Random Forest": {
            "r2": r2_score(y, y_pred_rf),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_rf))
            },
            "Gradient Boosting": {
            "r2": r2_score(y, y_pred_gb),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_gb))
            }
        }

       # Mostrar las métricas de evaluación del modelo
       with st.expander("**Métricas de Evaluación del Modelo:**"):
        for model, values in metrics.items():
            st.write(f"**{model}**")
            st.write(f"- Coeficiente de Determinación (R²): {values['r2']:.2f}")
            st.write(f"- Raíz del Error Cuadrático Medio (RMSE): {values['rmse']:.2f}")

       # Interfaz de usuario para predicciones
       st.write("### Predicción de Grasa Corporal")
       pbrazo_input = st.number_input("Ingresa el valor de PBrazo (cm): ", min_value=20.0, max_value=50.0, value=30.0, step=0.1)
       pcb_input = st.number_input("Ingresa el valor de PCB (mm): ", min_value=0.0, max_value=50.0, value=20.0, step=0.1)

       if st.button("Predecir Grasa Corporal"):
        prediccion_lr = modelo_grasa_lr.predict([[pbrazo_input, pcb_input]])[0]
        prediccion_dt = modelo_grasa_dt.predict([[pbrazo_input, pcb_input]])[0]
        prediccion_rf = modelo_grasa_rf.predict([[pbrazo_input, pcb_input]])[0]
        prediccion_gb = modelo_grasa_gb.predict([[pbrazo_input, pcb_input]])[0]

        st.write(f"### Predicciones de Grasa Corporal (%):")
        st.write(f"- **Regresión Lineal**: {prediccion_lr:.2f} %")
        st.write(f"- **Árbol de Decisión**: {prediccion_dt:.2f} %")
        st.write(f"- **Random Forest**: {prediccion_rf:.2f} %")
        st.write(f"- **Gradient Boosting**: {prediccion_gb:.2f} %")

############################################3333

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       import pandas as pd


       # Supongamos que 'df' es tu DataFrame ya cargado previamente
       data = df

       # Filtrar las columnas numéricas
       columnas_numericas = data.select_dtypes(include=[np.number]).columns

       # Seleccionar las variables independientes
       st.write("### Selección de Variables Independientes") 
       var_indep_1 = st.selectbox("Selecciona la primera variable independiente:", columnas_numericas)
       var_indep_2 = st.selectbox("Selecciona la segunda variable independiente:", columnas_numericas)

       X = data[[var_indep_1, var_indep_2]]

       y = data['Músculo (kg)']

       ppantorilla_range = np.linspace(X[var_indep_1].min(), X[var_indep_1].max(), 100)
       fa_range = np.linspace(X[var_indep_2].min(), X[var_indep_2].max(), 100)
       ppantorilla_grid, fa_grid = np.meshgrid(ppantorilla_range, fa_range)



       # Combina las características en una matriz bidimensional
       X_grid = np.c_[ppantorilla_grid.ravel(), fa_grid.ravel()]

       # Lista de modelos y sus nombres
       modelos = {
            'Regresión Lineal': LinearRegression(),
            'Árbol de Decisión': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

       # Entrenar y predecir con cada modelo
       predicciones = {}
       r2_scores = {}

       for nombre, modelo in modelos.items():
            modelo.fit(X, y)
            y_pred = modelo.predict(X_grid)
            y_pred = y_pred.reshape(ppantorilla_grid.shape)
            predicciones[nombre] = y_pred
            r2_scores[nombre] = modelo.score(X, y)

       # Crear pestañas en Streamlit
       tabs = st.tabs(list(modelos.keys()))

       for i, nombre in enumerate(modelos.keys()):
            with tabs[i]:
                st.header(f"{nombre}")
                y_pred = predicciones[nombre]
                fig = plt.figure(figsize=(10, 6))
                contour = plt.contourf(ppantorilla_grid, fa_grid, y_pred, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
                plt.scatter(X[var_indep_1], X[var_indep_2], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
                plt.xlabel(var_indep_1)
                plt.ylabel(var_indep_2)
                plt.title(f'Superficie de Decisión de {nombre} para Músculo (kg) (R^2={r2_scores[nombre]:.2f})')
                cbar = plt.colorbar(contour)
                cbar.set_label('Músculo (kg)')
                st.pyplot(fig)


       # Divide tus datos en características (X) y la variable dependiente (y) 
       X = data[[var_indep_1, var_indep_2]]
       y = data['Músculo (kg)']

       # Crear y ajustar los modelos 
       modelo_musculo_lr = LinearRegression()
       modelo_musculo_lr.fit(X, y)

       modelo_musculo_dt = DecisionTreeRegressor()
       modelo_musculo_dt.fit(X, y)

       modelo_musculo_rf = RandomForestRegressor()
       modelo_musculo_rf.fit(X, y)

       modelo_musculo_gb = GradientBoostingRegressor()
       modelo_musculo_gb.fit(X, y)

       # Realizar predicciones
       y_pred_lr = modelo_musculo_lr.predict(X)
       y_pred_dt = modelo_musculo_dt.predict(X)
       y_pred_rf = modelo_musculo_rf.predict(X)
       y_pred_gb = modelo_musculo_gb.predict(X)

       # Calcular el coeficiente de determinación (R^2) y RMSE para cada modelo
       metrics = {
        "Linear Regression": {
        "r2": r2_score(y, y_pred_lr),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_lr))
            },
        "Decision Tree": {
        "r2": r2_score(y, y_pred_dt),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_dt))
            },
        "Random Forest": {
        "r2": r2_score(y, y_pred_rf),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_rf))
            },
        "Gradient Boosting": {
        "r2": r2_score(y, y_pred_gb),
        "rmse": np.sqrt(mean_squared_error(y, y_pred_gb))
            }
        }

       # Mostrar las métricas de evaluación del modelo
       with st.expander("**Métricas de Evaluación del Modelo:**"):
        #st.write("### Métricas de Evaluación del Modelo:")
        for model, values in metrics.items():
            st.write(f"**{model}**")
            st.write(f"- Coeficiente de Determinación (R²): {values['r2']:.2f}")
            st.write(f"- Raíz del Error Cuadrático Medio (RMSE): {values['rmse']:.2f}")

       # Interfaz de usuario para predicciones
       st.write("### Predicción de Masa Muscular")
       var_indep_1_input = st.number_input(f"Ingresa el valor de {var_indep_1}: ", min_value=float(X[var_indep_1].min()), max_value=float(X[var_indep_1].max()), value=float(X[var_indep_1].mean()), step=0.1)
       var_indep_2_input = st.number_input(f"Ingresa el valor de {var_indep_2}: ", min_value=float(X[var_indep_2].min()), max_value=float(X[var_indep_2].max()), value=float(X[var_indep_2].mean()), step=0.1)
 

       if st.button("Predicción para la masa muscular (kg)"):
            prediccion_lr = modelo_musculo_lr.predict([[var_indep_1_input, var_indep_2_input]])[0]
            prediccion_dt = modelo_musculo_dt.predict([[var_indep_1_input, var_indep_2_input]])[0]
            prediccion_rf = modelo_musculo_rf.predict([[var_indep_1_input, var_indep_2_input]])[0]
            prediccion_gb = modelo_musculo_gb.predict([[var_indep_1_input, var_indep_2_input]])[0]

            st.write(f"### Predicciones de Músculo (kg):")
            st.write(f"- **Regresión Lineal**: {prediccion_lr:.2f} kg")
            st.write(f"- **Árbol de Decisión**: {prediccion_dt:.2f} kg")
            st.write(f"- **Random Forest**: {prediccion_rf:.2f} kg")
            st.write(f"- **Gradient Boosting**: {prediccion_gb:.2f} kg")



################################################

       import streamlit as st
       import numpy as np
       import matplotlib.pyplot as plt
       from matplotlib.colors import ListedColormap
       from sklearn.linear_model import LinearRegression
       from sklearn.tree import DecisionTreeRegressor
       from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
       from sklearn.metrics import mean_squared_error, r2_score
       import pandas as pd

       y = data['Grasa Corporal (%)']

       # Definir rangos de valores para las variables independientes
       var_indep_1_range = np.linspace(X[var_indep_1].min(), X[var_indep_1].max(), 100)
       var_indep_2_range = np.linspace(X[var_indep_2].min(), X[var_indep_2].max(), 100)
       var_indep_1_grid, var_indep_2_grid = np.meshgrid(var_indep_1_range, var_indep_2_range)

       # Combina las características en una matriz bidimensional
       X_grid = np.c_[var_indep_1_grid.ravel(), var_indep_2_grid.ravel()]

       # Lista de modelos y sus nombres
       modelos = {
        'Regresión Lineal': LinearRegression(),
        'Árbol de Decisión': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor()
       }

       # Entrenar y predecir con cada modelo
       predicciones = {}
       r2_scores = {}

       for nombre, modelo in modelos.items():
        modelo.fit(X, y)
        y_pred = modelo.predict(X_grid)
        y_pred = y_pred.reshape(var_indep_1_grid.shape)
        predicciones[nombre] = y_pred
        r2_scores[nombre] = modelo.score(X, y)

       # Crear pestañas en Streamlit
       tabs = st.tabs(list(modelos.keys()))

       for i, nombre in enumerate(modelos.keys()):
        with tabs[i]:
            st.header(f"{nombre}")
            y_pred = predicciones[nombre]
            fig = plt.figure(figsize=(10, 6))
            contour = plt.contourf(var_indep_1_grid, var_indep_2_grid, y_pred, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
            plt.scatter(X[var_indep_1], X[var_indep_2], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
            plt.xlabel(var_indep_1)
            plt.ylabel(var_indep_2)
            plt.title(f'Superficie de Decisión de {nombre} para Grasa Corporal (%) (R^2={r2_scores[nombre]:.2f})')
            cbar = plt.colorbar(contour)
            cbar.set_label('Grasa Corporal (%)')
            st.pyplot(fig)

# Divide tus datos en características (X) y la variable dependiente (y) 
#X = data[[var_indep_1, var_indep_2]]
#y = data['Grasa Corporal (%)']

       # Crear y ajustar los modelos 
       modelo_grasa_lr = LinearRegression()
       modelo_grasa_lr.fit(X, y)

       modelo_grasa_dt = DecisionTreeRegressor()
       modelo_grasa_dt.fit(X, y)
 
       modelo_grasa_rf = RandomForestRegressor()
       modelo_grasa_rf.fit(X, y)

       modelo_grasa_gb = GradientBoostingRegressor()
       modelo_grasa_gb.fit(X, y)

       # Realizar predicciones
       y_pred_lr = modelo_grasa_lr.predict(X)
       y_pred_dt = modelo_grasa_dt.predict(X)
       y_pred_rf = modelo_grasa_rf.predict(X)
       y_pred_gb = modelo_grasa_gb.predict(X)

       # Calcular el coeficiente de determinación (R^2) y RMSE para cada modelo
       metrics = {
         "Regresión Lineal": {
            "r2": r2_score(y, y_pred_lr),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_lr))
         },
        "Árbol de Decisión": {
            "r2": r2_score(y, y_pred_dt),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_dt))
         },
        "Random Forest": {
            "r2": r2_score(y, y_pred_rf),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_rf))
         },
        "Gradient Boosting": {
            "r2": r2_score(y, y_pred_gb),
            "rmse": np.sqrt(mean_squared_error(y, y_pred_gb))
         }
       }

       # Mostrar las métricas de evaluación del modelo
       with st.expander("**Métricas de Evaluación del Modelo:**"):
        for model, values in metrics.items():
            st.write(f"**{model}**")
            st.write(f"- Coeficiente de Determinación (R²): {values['r2']:.2f}")
            st.write(f"- Raíz del Error Cuadrático Medio (RMSE): {values['rmse']:.2f}")

       # Interfaz de usuario para predicciones
       st.write("### Predicción de Grasa Corporal")
       var_indep_1_input = st.number_input(f"Ingresa el valor para {var_indep_1}: ", min_value=float(X[var_indep_1].min()), max_value=float(X[var_indep_1].max()), value=float(X[var_indep_1].mean()), step=0.1)
       var_indep_2_input = st.number_input(f"Ingresa el valor para {var_indep_2}: ", min_value=float(X[var_indep_2].min()), max_value=float(X[var_indep_2].max()), value=float(X[var_indep_2].mean()), step=0.1)

       if st.button("Prediga el porcentaje de Grasa Corporal"):
        prediccion_lr = modelo_grasa_lr.predict([[var_indep_1_input, var_indep_2_input]])[0]
        prediccion_dt = modelo_grasa_dt.predict([[var_indep_1_input, var_indep_2_input]])[0]
        prediccion_rf = modelo_grasa_rf.predict([[var_indep_1_input, var_indep_2_input]])[0]
        prediccion_gb = modelo_grasa_gb.predict([[var_indep_1_input, var_indep_2_input]])[0]

        st.write(f"### Predicciones de Grasa Corporal (%):")
        st.write(f"- **Regresión Lineal**: {prediccion_lr:.2f} %")
        st.write(f"- **Árbol de Decisión**: {prediccion_dt:.2f} %")
        st.write(f"- **Random Forest**: {prediccion_rf:.2f} %")
        st.write(f"- **Gradient Boosting**: {prediccion_gb:.2f} %")





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
