# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from docxtpl import DocxTemplate
import os
import re
import time
import zipfile
from io import BytesIO

# --- Importaciones de Google Cloud (CORREGIDAS) ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Ensamblador de Fichas Técnicas con Vertex AI",
    page_icon="☁️",
    layout="wide"
)

# --- MODELOS DISPONIBLES ---
MODEL_OPTIONS = {
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
}

# --- FUNCIONES DE LÓGICA ---

def limpiar_html(texto_html):
    """Limpia etiquetas HTML de un texto."""
    if not isinstance(texto_html, str):
        return texto_html
    cleanr = re.compile('<.*?>')
    texto_limpio = re.sub(cleanr, '', texto_html)
    return texto_limpio

def setup_model(project_id, location, model_name):
    """Configura y retorna el cliente para el modelo Gemini en Vertex AI."""
    try:
        vertexai.init(project=project_id, location=location)
        
        generation_config = {
            "temperature": 0.6,
            "top_p": 1.0,
            "top_k": 32,
            "max_output_tokens": 8192,
        }
        
        # Forma correcta de definir las configuraciones de seguridad
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        model = GenerativeModel(
            model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        st.error(f"Error al inicializar Vertex AI: {e}")
        st.info("Asegúrate de haberte autenticado con 'gcloud auth application-default login' en tu terminal y de que las APIs necesarias estén habilitadas en tu proyecto de Google Cloud.")
        return None

# --- EJEMPLOS DE ALTA CALIDAD (FEW-SHOT PROMPTING) ---

EJEMPLOS_ANALISIS_PREMIUM = """
A continuación, te muestro ejemplos de análisis de la más alta calidad. Tu respuesta debe seguir este mismo estilo, tono y nivel de detalle.

### EJEMPLO 1: LECTURA LITERAL (TEXTO NARRATIVO) ###
**INSUMOS:**
- Competencia: Comprensión de textos
- Componente: Lectura literal
- Evidencia: Reconoce información específica en el texto.
- Enunciado: Los personajes del cuento son:
- Opciones: A: "Un hombre, un hombrecito y alguien que sostiene unas pinzas.", B: "Un narrador, un hombre y un hombrecito.", C: Un hombrecito y alguien que sostiene unas pinzas., D: Un hombre y el narrador.

**RESULTADO ESPERADO:**
Ruta Cognitiva Correcta:
Para responder la pregunta, el estudiante debe leer el cuento prestando atención a las entidades que realizan acciones o a quienes les suceden eventos en el texto. En el tercer párrafo, se menciona a "un hombre" que armó el barquito y a un "hombrecito diminuto" dentro de la botella. En el último párrafo, se describe que un "ojo enorme lo atisbaba desde fuera" al primer hombre y que "unas enormes pinzas que avanzaban hacia él". Este "ojo enorme" y las "enormes pinzas" implican la existencia de un tercer personaje, un ser que se encuentra mirando al primer personaje. El estudiante debe identificar a todos estos personajes que interactúan o son afectados por la trama.

Análisis de Opciones No Válidas:
- **Opción B:** No es correcta porque, en este cuento, el "narrador" es la voz que cuenta la historia, no un personaje que participe en los eventos del cuento. El relato está escrito en tercera persona y el narrador se mantiene fuera de la acción.
- **Opción C:** No es correcta porque omite al primer personaje introducido y central en la trama: "un hombre" que construye el barquito y observa al "hombrecito". Sin este personaje, la secuencia de eventos no se establece.
- **Opción D:** No es correcta porque, al igual que la opción B, incluye al "narrador" como personaje, lo cual es incorrecto. Además, omite al "hombrecito" y al ser con "unas pinzas", reduciendo el número de personajes activos en la historia.

### EJEMPLO 2: LECTURA INFERENCIAL (TEXTO NARRATIVO-INFORMATIVO) ###
**INSUMOS:**
- Competencia: Comprensión de textos
- Componente: Lectura inferencial
- Evidencia: Integra y compara diferentes partes del texto y analiza la estructura para hacer inferencias.
- Enunciado: Lee el siguiente fragmento del texto: “Los manglares están muriendo, por lo que el desequilibrio es cada vez mayor. La carretera lo cambió todo. Para construirla arrasaron veinte mil hectáreas de manglar...”. ¿Qué función cumple la parte subrayada dentro del fragmento?
- Opciones: A: Señalar la causa de un problema medioambiental., B: Establecer una comparación entre dos acciones de un proceso., C: Mostrar la consecuencia del daño medioambiental., D: Explicar el motivo por el que se decidió realizar una acción.

**RESULTADO ESPERADO:**
Ruta Cognitiva Correcta:
El estudiante debe comprender el contenido del fragmento y la estructura global del texto, para luego identificar cuál es la función que cumple dentro de esta. En este caso específico, el estudiante debe comprender que el fragmento señala la principal causa que ha llevado al desequilibrio del ecosistema de los manglares en la zona, y que este fragmento del texto justamente cumple con la función de señalar esa causa.

Análisis de Opciones No Válidas:
- **Opción B:** Es incorrecta porque la pregunta busca la causa del problema, no la comparación de acciones.
- **Opción C:** Es incorrecta porque el estudiante confunde la causa con la consecuencia del problema medioambiental. Identifica un efecto del problema, pero no su origen.
- **Opción D:** Es incorrecta porque se centra en la motivación detrás de una acción, en lugar de la causa del problema en sí mismo. La pregunta busca el origen del problema medioambiental.

### EJEMPLO 3: LECTURA CRÍTICA (TEXTO NARRATIVO-INFORMATIVO) ###
**INSUMOS:**
- Competencia: Comprensión de textos
- Componente: Lectura crítica
- Evidencia: Evalúa la credibilidad, confiabilidad y objetividad del texto, emitiendo juicios críticos sobre la información.
- Enunciado: ¿Por qué el autor cita el testimonio de Jesús Suárez en el texto?
- Opciones: A: Porque es el vocero que la comunidad palafítica ha designado., B: Porque es causante de la situación que ocurre en la población., C: Porque al ser experto en ecosistemas acuáticos su opinión es confiable., D: Porque al ser investigador puede verificar lo dicho por otro testigo de los hechos.

**RESULTADO ESPERADO:**
Ruta Cognitiva Correcta:
El estudiante analiza las opciones presentadas considerando la relación entre la justificación dada y la confiabilidad de la fuente. Evalúa la opción C y reconoce que la experticia en ecosistemas acuáticos otorga mayor credibilidad a la opinión de un individuo sobre una situación relacionada con este tema. Justifica la selección de la opción C al contrastarla con las demás opciones, considerando la relevancia de la experticia para la situación planteada.

Análisis de Opciones No Válidas:
- **Opción A:** Es incorrecta porque ser vocero no implica necesariamente tener el conocimiento experto para opinar sobre situaciones específicas.
- **Opción B:** Es incorrecta porque ser causante de un problema no implica tener el conocimiento o la imparcialidad para analizarlo y ofrecer una opinión confiable.
- **Opción D:** Es incorrecta porque la verificación de un testimonio en este contexto requiere una experticia específica en el tema, que en este caso es ecosistemas acuáticos.
"""

EJEMPLOS_RECOMENDACIONES_PREMIUM = """
A continuación, te muestro ejemplos de recomendaciones pedagógicas de la más alta calidad. Tu respuesta debe seguir este mismo estilo, estructura y enfoque creativo.

### EJEMPLO 1 DE RECOMENDACIONES PERFECTAS (TEXTO DISCONTINUO) ###
**INSUMOS:**
- Qué Evalúa la pregunta: la pregunta evalúa la habilidad del estudiante para relacionar diferentes elementos del contenido e identificar nueva información en textos no literarios.
- Evidencia: Relaciona diferentes partes del texto para hacer inferencias sobre significados o sobre el propósito general.

**RESULTADO ESPERADO:**
RECOMENDACIÓN PARA FORTALECER EL APRENDIZAJE EVALUADO EN la pregunta
Para reforzar la habilidad de vincular diferentes elementos del contenido y descubrir nuevas ideas, se sugiere la realización de actividades que impliquen el análisis de textos no literarios de carácter discontinuo como infografías. Los estudiantes podrían empezar por leer estas fuentes y marcar los datos que consideren relevantes. Posteriormente, en un esfuerzo colectivo, podrían construir un mapa conceptual que refleje la relación entre los diferentes datos resaltados. Finalmente, podrían trabajar en la identificación de las ideas principales y secundarias que emergen de este mapa, lo que les permitirá tener una comprensión más profunda del texto.

RECOMENDACIÓN PARA AVANZAR EN EL APRENDIZAJE EVALUADO EN la pregunta
Para consolidar la capacidad de identificar las funciones de los diferentes elementos que componen un texto no literario de carácter discontinuo, se sugiere fomentar la práctica de reorganizar textos desordenados. Los estudiantes pueden recibir fragmentos de una infografía que deben arreglar en el orden correcto, identificando la introducción, el desarrollo y la conclusión. Durante esta actividad, se pueden formular preguntas como: ¿Cuál fragmento introduce el tema? ¿Qué información proporciona esta imagen o gráfico? ¿Cómo se relaciona con el texto?

### EJEMPLO 2 DE RECOMENDACIONES PERFECTAS (TEXTO INFORMATIVO) ###
**INSUMOS:**
- Qué Evalúa la pregunta: Este ítem evalúa la capacidad del estudiante para hacer una inferencia integrando información implícita presente en una parte del texto.
- Evidencia: Integra y compara diferentes partes del texto y analiza la estructura para hacer inferencias.

**RESULTADO ESPERADO:**
RECOMENDACIÓN PARA FORTALECER EL APRENDIZAJE EVALUADO EN la pregunta
Para fortalecer la habilidad de hacer inferencias a partir de un segmento de un texto informativo, se sugiere implementar una dinámica de "lectura de pistas". Esta estrategia se enfoca en que los estudiantes identifiquen información implícita en fragmentos textuales cortos para inferir contextos o emociones que no se mencionan directamente. El docente puede presentar al grupo tres o cuatro fragmentos muy breves y evocadores (de noticias o crónicas) que insinúen una situación sin describirla por completo. Por ejemplo: "El teléfono sonó por décima vez. Al otro lado de la línea, solo se oía una respiración agitada. Afuera, la sirena de una ambulancia se acercaba". Los estudiantes, en parejas, leen el fragmento y discuten qué pueden deducir de la escena. Las preguntas orientadoras pueden ser: ¿Qué pistas te da el texto sobre el estado de ánimo de la persona?, ¿Qué crees que pasó justo antes de la escena descrita?

RECOMENDACIÓN PARA AVANZAR EN EL APRENDIZAJE EVALUADO EN la pregunta
Para avanzar en la habilidad de hacer inferencias complejas a partir de la comparación de diferentes partes de un texto, se sugiere proponer un análisis de perspectivas múltiples dentro de una misma crónica o texto informativo. El objetivo es que los estudiantes superen la inferencia local y aprendan a contrastar voces, datos o argumentos presentados en un mismo relato. El docente puede seleccionar una crónica periodística sobre un tema urbano actual que incluya las voces de distintos actores sociales (un vendedor, un residente, un funcionario). Los estudiantes deben leer el texto e identificar y comparar las diferentes posturas frente al mismo hecho. Las preguntas orientadoras pueden ser: ¿Qué similitudes y diferencias encuentras entre las perspectivas?, ¿Qué visión del problema se formaría un lector si el texto solo hubiera incluido una de estas voces?
"""

# --- FUNCIONES DE PROMPTS SECUENCIALES ---

def construir_prompt_paso1_analisis_central(fila, instruccion_adicional=""):
    """Paso 1: Genera la Ruta Cognitiva y el Análisis de Distractores, guiado por ejemplos."""
    fila = fila.fillna('')
    descripcion_item = (
        f"Enunciado: {fila.get('Enunciado', '')}\n"
        f"A. {fila.get('OpcionA', '')}\n"
        f"B. {fila.get('OpcionB', '')}\n"
        f"C. {fila.get('OpcionC', '')}\n"
        f"D. {fila.get('OpcionD', '')}\n"
        f"Respuesta correcta: {fila.get('AlternativaClave', '')}"
    )
    instruccion_formateada = f"\n**Instrucción Adicional del Usuario:** {instruccion_adicional}\n" if instruccion_adicional else ""
    return f"""
🎯 ROL DEL SISTEMA
Eres un experto psicómetra y pedagogo. Tu misión es deconstruir un ítem de evaluación siguiendo el estilo y la calidad de los ejemplos proporcionados.

{EJEMPLOS_ANALISIS_PREMIUM}

🧠 INSUMOS DE ENTRADA (Para el nuevo ítem que debes analizar):
- Texto/Fragmento: {fila.get('ItemContexto', 'No aplica')}
- Descripción dla pregunta: {fila.get('ItemEnunciado', 'No aplica')}
- Componente: {fila.get('ComponenteNombre', 'No aplica')}
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}
- Tipología Textual (Solo para Lectura Crítica): {fila.get('Tipologia Textual', 'No aplica')}
- Grado Escolar: {fila.get('ItemGradoId', '')}
- Análisis de Errores Comunes: {fila.get('Analisis_Errores', 'No aplica')}
- Respuesta correcta: {fila.get('AlternativaClave', 'No aplica')}
- Opción A: {fila.get('OpcionA', 'No aplica')}
- Opción B: {fila.get('OpcionB', 'No aplica')}
- Opción C: {fila.get('OpcionC', 'No aplica')}
- Opción D: {fila.get('OpcionD', 'No aplica')}


📝 INSTRUCCIONES
Basándote en los ejemplos de alta calidad y los nuevos insumos, realiza el siguiente proceso en dos fases:
{instruccion_formateada}

FASE 1: RUTA COGNITIVA
Describe, en un párrafo continuo y de forma impersonal, el procedimiento mental que un estudiante debe ejecutar para llegar a la respuesta correcta.
1.  **Genera la Ruta Cognitiva:** Describe el paso a paso mental y lógico que un estudiante debe seguir para llegar a la respuesta correcta. Usa verbos que representen procesos cognitivos.
2.  **Auto-Verificación:** Revisa que la ruta se alinee con la Competencia ('{fila.get('CompetenciaNombre', '')}') y la Evidencia ('{fila.get('EvidenciaNombre', '')}').
3.  **Justificación Final:** El último paso debe justificar la elección de la respuesta correcta.

FASE 2: ANÁLISIS DE OPCIONES NO VÁLIDAS
- Para cada opción incorrecta, identifica la naturaleza del error y explica el razonamiento fallido.
- Luego, explica el posible razonamiento que lleva al estudiante a cometer ese error.
- Finalmente, clarifica por qué esa opción es incorrecta en el contexto de la tarea evaluativa.

✍️ FORMATO DE SALIDA
**REGLA CRÍTICA:** Responde únicamente con los dos títulos siguientes, en este orden y sin añadir texto adicional.

Ruta Cognitiva Correcta:
[Párrafo continuo y detallado.] Debe describir como es la secuencia de procesos cognitivos. Ejemplo: Para resolver correctamente este ítem, el estudiante primero debe [verbo cognitivo 1]... Luego, necesita [verbo cognitivo 2]... Este proceso le permite [verbo cognitivo 3]..., lo que finalmente lo lleva a concluir que la opción [letra de la respuesta correcta] es la correcta porque [justificación final].

Análisis de Opciones No Válidas:
- **Opción [Letra del distractor]:** El estudiante podría escoger esta opción si comete un error de [naturaleza de la confusión u error], lo que lo lleva a pensar que [razonamiento erróneo]. Sin embargo, esto es incorrecto porque [razón clara y concisa].
"""
    
def construir_prompt_paso2_sintesis_que_evalua(analisis_central_generado, fila, instruccion_adicional=""):
    """Paso 2: Sintetiza el "Qué Evalúa" a partir del análisis central."""
    fila = fila.fillna('')
    try:
        header_distractores = "Análisis de Opciones No Válidas:"
        idx_distractores = analisis_central_generado.find(header_distractores)
        ruta_cognitiva_texto = analisis_central_generado[:idx_distractores].strip() if idx_distractores != -1 else analisis_central_generado
    except:
        ruta_cognitiva_texto = analisis_central_generado
    instruccion_formateada = f"\n**Instrucción Adicional del Usuario:** {instruccion_adicional}\n" if instruccion_adicional else ""
    return f"""
🎯 ROL DEL SISTEMA
Eres un experto en evaluación que sintetiza análisis complejos en una sola frase concisa.

🧠 INSUMOS DE ENTRADA
A continuación, te proporciono un análisis detallado de la ruta cognitiva necesaria para resolver un ítem.

ANÁLISIS DE LA RUTA COGNITIVA:
---
{ruta_cognitiva_texto}
---

TAXONOMÍA DE REFERENCIA:
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}

📝 INSTRUCCIONES
{instruccion_formateada}
Basándote **exclusivamente** en el ANÁLISIS DE LA RUTA COGNITIVA, redacta una única frase (máximo 2 renglones) que resuma la habilidad principal que se está evaluando.
- **Regla 1:** La frase debe comenzar obligatoriamente con "Este ítem evalúa la capacidad del estudiante para...".
- **Regla 2:** La frase debe describir los **procesos cognitivos**, no debe contener especificamene ninguno de los elementos del texto o dla pregunta, busca en cambio palabras/expresiones genéricas en reemplazo de elementos del item/texto cuando es necesario.
- **Regla 3:** Utiliza la TAXONOMÍA DE REFERENCIA para asegurar que el lenguaje sea preciso y alineado.

✍️ FORMATO DE SALIDA
Responde únicamente con la frase solicitada, sin el título "Qué Evalúa".
"""

def construir_prompt_paso3_recomendaciones(que_evalua_sintetizado, analisis_central_generado, fila, instruccion_adicional=""):
    """Paso 3: Genera las recomendaciones, guiado por ejemplos."""
    fila = fila.fillna('')
    instruccion_formateada = f"\n**Instrucción Adicional del Usuario:** {instruccion_adicional}\n" if instruccion_adicional else ""
    return f"""
🎯 ROL DEL SISTEMA
Eres un diseñador instruccional experto y un docente de aula con mucha experiencia. Tu especialidad es crear actividades de lectura que son novedosas, lúdicas y, sobre todo, prácticas y realizables en un salón de clases con recursos limitados.

{EJEMPLOS_RECOMENDACIONES_PREMIUM}

🧠 INSUMOS DE ENTRADA (Para el nuevo ítem):
# Se mantienen los insumos para dar contexto, pero las instrucciones forzarán a la IA a no usarlos literalmente.
- Qué Evalúa la pregunta: {que_evalua_sintetizado}
- Análisis Detallado dla pregunta: {analisis_central_generado}
- Texto/Fragmento: {fila.get('ItemContexto', 'No aplica')}
- Descripción dla pregunta: {fila.get('ItemEnunciado', 'No aplica')}
- Componente: {fila.get('ComponenteNombre', 'No aplica')}
- Competencia: {fila.get('CompetenciaNombre', '')}
- Aprendizaje Priorizado: {fila.get('AfirmacionNombre', '')}
- Evidencia de Aprendizaje: {fila.get('EvidenciaNombre', '')}
- Tipología Textual (Solo para Lectura Crítica): {fila.get('Tipologia Textual', 'No aplica')}
- Grado Escolar: {fila.get('ItemGradoId', '')}
- Análisis de Errores Comunes: {fila.get('Analisis_Errores', 'No aplica')}
- Respuesta correcta: {fila.get('AlternativaClave', 'No aplica')}

📝 INSTRUCCIONES PARA GENERAR LAS RECOMENDACIONES
{instruccion_formateada}
Basándote en los ejemplos y los insumos, genera dos recomendaciones (Fortalecer y Avanzar) que cumplan con estas reglas inviolables:
1.  **ABSTRACCIÓN DE LA HABILIDAD:** # <-- CAMBIO CLAVE 1: Desanclar del ítem.
    Las actividades deben enfocarse en la habilidad cognitiva descrita en 'Qué Evalúa la pregunta', no en el contenido específico del 'Texto/Fragmento' o la 'Descripción dla pregunta'. Usa los insumos solo para entender la habilidad, pero diseña una actividad que se pueda aplicar a OTROS textos o contextos.
    CRÍTICO: Evita usar las mismas situaciones expuestas en el ítem. Deben ser diferentes pero debene estar dentro del mismo campo cognitivo de lo que evalúa el ítem.
    
2.  **VIABILIDAD EN EL AULA:** # <-- CAMBIO CLAVE 2: Realismo y practicidad.
    Las actividades deben ser 100% realizables dentro de un salón de clases estándar. Esto significa:
    - **Cero Materiales:** No requieren preparación de materiales especiales (fichas, tarjetas, proyectores). Se basan en la discusión, el análisis oral o la interacción con un texto genérico.
    - **Cero Tareas Externas:** No implican actividades fuera del aula, ni uso de tecnología.
    - **La novedad está en la dinámica, no en los recursos.**

3.  **CERO PRODUCCIÓN ESCRITA:** Deben ser actividades exclusivas de lectura, selección, debate corto, clasificación oral o argumentación.

4.  **CREATIVIDAD BASADA EN EL ESCENARIO:** # <-- ESTA ES LA REGLA CLAVE MODIFICADA
    La novedad y el factor lúdico deben residir en la **situación o el contexto** que se plantea, no necesariamente en el formato de la interacción.
    - **EVITA** escenarios clichés o abstractos típicos de libros de texto. Por ejemplo, en lugar de "lanzar un dado 20 veces", que es un escenario aburrido...
    - **FAVORECE** escenarios concretos, imaginativos y con una narrativa. Por ejemplo, para la misma habilidad de probabilidad, podrías proponer: "se organiza un pequeño 'casino' en clase con barajas de colores donde los estudiantes deben calcular sus posibilidades de ganar en diferentes juegos inventados". O "los estudiantes son exploradores que deben decidir qué camino tomar en una jungla basándose en las probabilidades de encontrar recursos o peligros".
    - El objetivo es crear un "mini-mundo" o un reto temático donde se aplique la habilidad.

5.  **REDACCIÓN IMPERSONAL.**

RECOMENDACIÓN PARA FORTALECER EL APRENDIZAJE EVALUADO EN la pregunta
Para fortalecer la habilidad de [verbo clave extraído de la Evidencia de Aprendizaje], se sugiere [descripción de la estrategia de andamiaje para ese proceso exacto].
Una actividad que se puede hacer es: [Descripción detallada de la actividad novedosa y creativa, que no implica escritura].
Las preguntas orientadoras para esta actividad, entre otras, pueden ser:
- [Pregunta 1: Que guíe el primer paso del proceso cognitivo]
- [Pregunta 2: Que ayude a analizar un componente clave del proceso]
- [Pregunta 3: Que conduzca a la conclusión del proceso base]

RECOMENDACIÓN PARA AVANZAR EN EL APRENDIZAJE EVALUADO EN la pregunta
Para avanzar desde [proceso cognitivo de Fortalecer] hacia la habilidad de [verbo clave del proceso cognitivo superior], se sugiere [descripción de la estrategia de complejización].
Una actividad que se puede hacer es: [Descripción detallada de la actividad estimulante y poco convencional, que no implique escritura].
Las preguntas orientadoras para esta actividad, entre otras, pueden ser:
- [Pregunta 1: De análisis o evaluación que requiera un razonamiento más profundo]
- [Pregunta 2: De aplicación, comparación o transferencia a un nuevo contexto]
- [Pregunta 3: De metacognición o pensamiento crítico sobre el proceso completo]
"""

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---

st.title("☁️ Ensamblador de Fichas Técnicas con Vertex AI")
st.markdown("Una aplicación para enriquecer datos pedagógicos usando los modelos de Google Cloud.")

if 'df_enriquecido' not in st.session_state:
    st.session_state.df_enriquecido = None
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None

# --- PASO 0: Configuración de Google Cloud en la Barra Lateral ---
st.sidebar.header("☁️ Configuración de Google Cloud")

project_id = st.sidebar.text_input(
    "Ingresa tu ID de Proyecto de Google Cloud",
    value=os.environ.get("GCP_PROJECT_ID", ""),
    help="El identificador único de tu proyecto en GCP."
)
location = st.sidebar.text_input(
    "Ingresa la Región de GCP",
    value=os.environ.get("GCP_LOCATION", "us-central1"),
    help="Ejemplo: us-central1, europe-west2, etc."
)
selected_model_key = st.sidebar.selectbox(
    "Elige el modelo de Gemini a utilizar",
    options=list(MODEL_OPTIONS.keys()),
    help="Gemini 2.5 Pro es más potente, mientras que Flash es más rápido y económico."
)

with st.sidebar.expander("ℹ️ ¿Cómo funciona la autenticación?"):
    st.write("""
    Esta aplicación utiliza **Application Default Credentials (ADC)** para autenticarse con Google Cloud.
    
    **Si ejecutas esto en tu PC local:**
    1. Instala la CLI de Google Cloud (`gcloud`).
    2. Ejecuta el siguiente comando en tu terminal:
       ```bash
       gcloud auth application-default login
       ```
    3. Sigue las instrucciones para iniciar sesión con tu cuenta de Google.
    
    **Si despliegas esta aplicación (ej. en Cloud Run):**
    El entorno gestionado se encargará de la autenticación automáticamente a través de la cuenta de servicio asociada.
    """)

# --- PASO 1: Carga de Archivos ---
st.header("Paso 1: Carga tus Archivos")
col1, col2 = st.columns(2)
with col1:
    archivo_excel = st.file_uploader("Sube tu Excel con los datos base", type=["xlsx"])
with col2:
    archivo_plantilla = st.file_uploader("Sube tu Plantilla de Word", type=["docx"])

# --- PASO 2: Enriquecimiento con IA ---    
st.header("Paso 2: Enriquece tus Datos con IA")
# AÑADE ESTE BLOQUE
with st.expander("💡 Opcional: Añadir Instrucciones Adicionales a la IA"):
    st.markdown("Usa estos campos para guiar o refinar el trabajo de la IA en cada paso.")
    instruccion_paso1 = st.text_area(
        "Instrucciones para el Paso 1 (Análisis Central)",
        placeholder="Ej: Presta especial atención a la ironía en el texto.",
        help="Guía para la Ruta Cognitiva y el Análisis de Distractores."
    )
    instruccion_paso2 = st.text_area(
        "Instrucciones para el Paso 2 (Síntesis 'Qué Evalúa')",
        placeholder="Ej: Asegúrate de que la síntesis use el verbo 'interpretar'.",
        help="Guía para la frase que resume la habilidad evaluada."
    )
    instruccion_paso3 = st.text_area(
        "Instrucciones para el Paso 3 (Recomendaciones)",
        placeholder="Ej: Orienta las recomendaciones hacia un enfoque colaborativo.",
        help="Guía para el diseño de las actividades de Fortalecer y Avanzar."
    )

if st.button("🤖 Iniciar Análisis y Generación", disabled=(not project_id or not location or not archivo_excel)):
    if not project_id or not location:
        st.error("Por favor, completa la configuración de Google Cloud en la barra lateral izquierda.")
    elif not archivo_excel:
        st.warning("Por favor, sube un archivo Excel para continuar.")
    else:
        model_name = MODEL_OPTIONS[selected_model_key]
        model = setup_model(project_id, location, model_name)
        
        if model:
            st.success(f"Conectado a Vertex AI en el proyecto '{project_id}' usando el modelo '{model_name}'.")
            with st.spinner("Procesando archivo Excel y preparando datos..."):
                df = pd.read_excel(archivo_excel)
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(limpiar_html)

                columnas_nuevas = ["Que_Evalua", "Justificacion_Correcta", "Analisis_Distractores", "Recomendacion_Fortalecer", "Recomendacion_Avanzar"]
                for col in columnas_nuevas:
                    if col not in df.columns:
                        df[col] = ""
                st.success("Datos limpios y listos.")

            progress_bar_main = st.progress(0, text="Iniciando Proceso...")
            total_filas = len(df)

            for i, fila in df.iterrows():
                item_id = fila.get('ItemId', i + 1)
                st.markdown(f"--- \n ### Procesando Ítem: **{item_id}**")
                progress_bar_main.progress(i / total_filas, text=f"Procesando ítem {i+1}/{total_filas}")

                with st.container(border=True):
                    try:
                        # --- LLAMADA 1: ANÁLISIS CENTRAL ---
                        st.write(f"**Paso 1/3:** Realizando análisis central...")
                        # Pasa el contenido de la caja de texto a la función
                        prompt_paso1 = construir_prompt_paso1_analisis_central(fila, instruccion_paso1) 
                        response_paso1 = model.generate_content(prompt_paso1)
                        analisis_central = response_paso1.text.strip()
                        time.sleep(1) 

                        header_correcta = "Ruta Cognitiva Correcta:"
                        header_distractores = "Análisis de Opciones No Válidas:"
                        idx_distractores = analisis_central.find(header_distractores)
                        
                        if idx_distractores == -1:
                            raise ValueError("La respuesta de la IA (Paso 1) no contiene el separador 'Análisis de Opciones No Válidas'.")

                        ruta_cognitiva = analisis_central[len(header_correcta):idx_distractores].strip()
                        analisis_distractores = analisis_central[idx_distractores:].strip()

                        # --- LLAMADA 2: SÍNTESIS DEL "QUÉ EVALÚA" ---
                        st.write(f"**Paso 2/3:** Sintetizando 'Qué Evalúa'...")
                        # Pasa el contenido de la caja de texto a la función
                        prompt_paso2 = construir_prompt_paso2_sintesis_que_evalua(analisis_central, fila, instruccion_paso2)
                        response_paso2 = model.generate_content(prompt_paso2)
                        que_evalua = response_paso2.text.strip()
                        time.sleep(1)
                        
                        # --- LLAMADA 3: GENERACIÓN DE RECOMENDACIONES ---
                        st.write(f"**Paso 3/3:** Generando recomendaciones...")
                        # Pasa el contenido de la caja de texto a la función
                        prompt_paso3 = construir_prompt_paso3_recomendaciones(que_evalua, analisis_central, fila, instruccion_paso3)
                        response_paso3 = model.generate_content(prompt_paso3)
                        recomendaciones = response_paso3.text.strip()
                        
                        titulo_avanzar = "RECOMENDACIÓN PARA AVANZAR"
                        idx_avanzar = recomendaciones.upper().find(titulo_avanzar)
                        
                        if idx_avanzar == -1:
                             raise ValueError("La respuesta de la IA (Paso 3) no contiene el separador 'RECOMENDACIÓN PARA AVANZAR'.")

                        fortalecer = recomendaciones[:idx_avanzar].strip()
                        avanzar = recomendaciones[idx_avanzar:].strip()

                        # --- GUARDAR TODO EN EL DATAFRAME ---
                        df.loc[i, "Que_Evalua"] = que_evalua
                        df.loc[i, "Justificacion_Correcta"] = ruta_cognitiva
                        df.loc[i, "Analisis_Distractores"] = analisis_distractores
                        df.loc[i, "Recomendacion_Fortalecer"] = fortalecer
                        df.loc[i, "Recomendacion_Avanzar"] = avanzar
                        st.success(f"Ítem {item_id} procesado con éxito.")

                    except Exception as e:
                        st.error(f"Ocurrió un error procesando la pregunta {item_id}: {e}")
                        df.loc[i, "Que_Evalua"] = "ERROR EN PROCESAMIENTO"
                        # Puedes agregar más detalles del error si lo necesitas
                        df.loc[i, "Justificacion_Correcta"] = f"Error: {e}" 
            
            progress_bar_main.progress(1.0, text="¡Proceso completado!")
            st.session_state.df_enriquecido = df
            st.balloons()
        else:
            st.error("No se pudo inicializar el modelo de IA. Verifica tu configuración de GCP.")

# --- PASO 3: Vista Previa y Descarga de Excel ---
if st.session_state.df_enriquecido is not None:
    st.header("Paso 3: Verifica y Descarga los Datos Enriquecidos")
    st.dataframe(st.session_state.df_enriquecido.head())
    
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.df_enriquecido.to_excel(writer, index=False, sheet_name='Datos Enriquecidos')
    output_excel.seek(0)
    
    st.download_button(
        label="📥 Descargar Excel Enriquecido",
        data=output_excel,
        file_name="excel_enriquecido_con_ia.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- PASO 4: Ensamblaje y Descarga de Fichas ---
if st.session_state.df_enriquecido is not None and archivo_plantilla is not None:
    st.header("Paso 4: Ensambla y Descarga las Fichas Técnicas")
    
    columna_nombre_archivo = st.text_input(
        "Escribe el nombre de la columna para nombrar los archivos (ej. ItemId)",
        value="ItemId"
    )
    
    if st.button("📄 Ensamblar Fichas Técnicas", type="primary"):
        df_final = st.session_state.df_enriquecido
        if columna_nombre_archivo not in df_final.columns:
            st.error(f"La columna '{columna_nombre_archivo}' no existe en el Excel. Elige una de: {', '.join(df_final.columns)}")
        else:
            with st.spinner("Ensamblando todas las fichas en un archivo .zip..."):
                plantilla_bytes = BytesIO(archivo_plantilla.getvalue())
                zip_buffer = BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    total_docs = len(df_final)
                    progress_bar_zip = st.progress(0, text="Iniciando ensamblaje...")
                    for i, fila in df_final.iterrows():
                        plantilla_bytes.seek(0) # ¡Importante! Reinicia el buffer de la plantilla
                        doc = DocxTemplate(plantilla_bytes)
                        contexto = fila.to_dict()
                        contexto_limpio = {k: (v if pd.notna(v) else "") for k, v in contexto.items()}
                        doc.render(contexto_limpio)
                        
                        doc_buffer = BytesIO()
                        doc.save(doc_buffer)
                        doc_buffer.seek(0)
                        
                        nombre_base = str(fila.get(columna_nombre_archivo, f"ficha_{i+1}")).replace('/', '_').replace('\\', '_')
                        nombre_archivo_salida = f"{nombre_base}.docx"
                        
                        zip_file.writestr(nombre_archivo_salida, doc_buffer.getvalue())
                        progress_bar_zip.progress((i + 1) / total_docs, text=f"Añadiendo ficha {i+1}/{total_docs} al .zip")
                
                st.session_state.zip_buffer = zip_buffer
                st.success("¡Ensamblaje completado!")

if st.session_state.zip_buffer:
    st.download_button(
        label="📥 Descargar TODAS las fichas (.zip)",
        data=st.session_state.zip_buffer.getvalue(),
        file_name="fichas_tecnicas_generadas.zip",
        mime="application/zip"
    )
