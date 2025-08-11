# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from docxtpl import DocxTemplate
import os
import re
import time
import zipfile
from io import BytesIO

# --- Importaciones de Google Cloud ---
import vertexai
from google.cloud import storage
from vertexai.preview.generative_models import GenerativeModel, Part

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Ensamblador de Fichas Técnicas Google Vertex AI",
    page_icon="🤖",
    layout="wide"
)

# --- VARIABLES DE ENTORNO ---
# Estas variables se configurarán en tu entorno de despliegue (Cloud Run).
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION")
GCP_STORAGE_BUCKET = os.environ.get("GCP_STORAGE_BUCKET")

# --- FUNCIONES DE LÓGICA ---

def limpiar_html(texto_html):
    """Limpia etiquetas HTML de un texto."""
    if not isinstance(texto_html, str):
        return texto_html
    cleanr = re.compile('<.*?>')
    texto_limpio = re.sub(cleanr, '', texto_html)
    return texto_limpio

def leer_prompt_desde_gcs(nombre_archivo):
    """Lee el contenido de un archivo de prompt desde Cloud Storage."""
    if not GCP_STORAGE_BUCKET:
        st.error("Error: La variable de entorno 'GCP_STORAGE_BUCKET' no está configurada.")
        return None
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCP_STORAGE_BUCKET)
        blob = bucket.blob(nombre_archivo)
        
        if not blob.exists():
            st.error(f"Error: El archivo de prompt '{nombre_archivo}' no se encontró en el bucket '{GCP_STORAGE_BUCKET}'.")
            return None

        # Intenta descargar y leer el contenido del archivo
        contenido_prompt = blob.download_as_text()
        return contenido_prompt
    except Exception as e:
        # ---- ESTA ES LA MODIFICACIÓN CLAVE ----
        # Si algo falla al intentar leer, mostramos el error técnico exacto.
        st.error(f"Error al LEER el archivo '{nombre_archivo}'. Causa raíz:")
        st.error(f"Error detallado: {e}")
        return None

def subir_a_cloud_storage(data_buffer, file_name, content_type):
    """Sube un archivo de un buffer a un bucket de Cloud Storage."""
    if not GCP_STORAGE_BUCKET:
        st.error("Error: La variable de entorno 'GCP_STORAGE_BUCKET' no está configurada.")
        return None
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCP_STORAGE_BUCKET)
        blob = bucket.blob(file_name)
        
        data_buffer.seek(0)
        blob.upload_from_file(data_buffer, content_type=content_type)
        
        st.success(f"Archivo subido a Cloud Storage: gs://{GCP_STORAGE_BUCKET}/{file_name}")
        return f"gs://{GCP_STORAGE_BUCKET}/{file_name}"
    except Exception as e:
        st.error(f"Error al subir el archivo a Cloud Storage: {e}")
        return None

def construir_prompt_paso1_analisis_central(fila, prompt_template):
    """Paso 1: Genera la Ruta Cognitiva y el Análisis de Distractores, guiado por un prompt externo."""
    fila = fila.fillna('')
    
    return prompt_template.format(
        ItemContexto=fila.get('ItemContexto', 'No aplica'),
        ItemEnunciado=fila.get('ItemEnunciado', 'No aplica'),
        ComponenteNombre=fila.get('ComponenteNombre', 'No aplica'),
        CompetenciaNombre=fila.get('CompetenciaNombre', ''),
        AfirmacionNombre=fila.get('AfirmacionNombre', ''),
        EvidenciaNombre=fila.get('EvidenciaNombre', ''),
        Tipologia_Textual=fila.get('Tipologia Textual', 'No aplica'),
        ItemGradoId=fila.get('ItemGradoId', ''),
        Analisis_Errores=fila.get('Analisis_Errores', 'No aplica'),
        AlternativaClave=fila.get('AlternativaClave', 'No aplica'),
        OpcionA=fila.get('OpcionA', 'No aplica'),
        OpcionB=fila.get('OpcionB', 'No aplica'),
        OpcionC=fila.get('OpcionC', 'No aplica'),
        OpcionD=fila.get('OpcionD', 'No aplica')
    )

def construir_prompt_paso2_sintesis_que_evalua(analisis_central_generado, fila, prompt_template):
    """Paso 2: Sintetiza el "Qué Evalúa" a partir del análisis central, guiado por un prompt externo."""
    fila = fila.fillna('')
    try:
        header_distractores = "Análisis de Opciones No Válidas:"
        idx_distractores = analisis_central_generado.find(header_distractores)
        ruta_cognitiva_texto = analisis_central_generado[:idx_distractores].strip() if idx_distractores != -1 else analisis_central_generado
    except:
        ruta_cognitiva_texto = analisis_central_generado

    return prompt_template.format(
        ruta_cognitiva_texto=ruta_cognitiva_texto,
        CompetenciaNombre=fila.get('CompetenciaNombre', ''),
        AfirmacionNombre=fila.get('AfirmacionNombre', ''),
        EvidenciaNombre=fila.get('EvidenciaNombre', '')
    )

def construir_prompt_paso3_recomendaciones(que_evalua_sintetizado, analisis_central_generado, fila, prompt_template):
    """Paso 3: Genera las recomendaciones, guiado por un prompt externo."""
    fila = fila.fillna('')
    return prompt_template.format(
        que_evalua_sintetizado=que_evalua_sintetizado,
        analisis_central_generado=analisis_central_generado,
        ItemContexto=fila.get('ItemContexto', 'No aplica'),
        ItemEnunciado=fila.get('ItemEnunciado', 'No aplica'),
        ComponenteNombre=fila.get('ComponenteNombre', 'No aplica'),
        CompetenciaNombre=fila.get('CompetenciaNombre', ''),
        AfirmacionNombre=fila.get('AfirmacionNombre', ''),
        EvidenciaNombre=fila.get('EvidenciaNombre', ''),
        Tipologia_Textual=fila.get('Tipologia Textual', 'No aplica'),
        ItemGradoId=fila.get('ItemGradoId', ''),
        Analisis_Errores=fila.get('Analisis_Errores', 'No aplica'),
        AlternativaClave=fila.get('AlternativaClave', 'No aplica')
    )

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---
st.title("🤖 Ensamblador de Fichas Técnicas con Google Vertex IA")
st.markdown("Una aplicación para enriquecer datos pedagógicos y generar fichas personalizadas.")

if 'df_enriquecido' not in st.session_state:
    st.session_state.df_enriquecido = None
if 'zip_buffer' not in st.session_state:
    st.session_state.zip_buffer = None
if 'prompts_cache' not in st.session_state:
    st.session_state.prompts_cache = {}

# --- PASO 0: Configuración y Validación ---
st.sidebar.header("🔑 Configuración")
st.info("Esta aplicación usa Google Cloud Storage para leer los prompts y guardar los resultados.")

# --- PANEL DE DIAGNÓSTICO ---
with st.sidebar.expander("🔍 Panel de Diagnóstico de Sistema", expanded=True):
    st.write("Verificando la configuración y el acceso a los prompts...")

    # 1. Verificar Variables de Entorno
    st.subheader("1. Variables de Entorno")
    bucket_name = os.environ.get("GCP_STORAGE_BUCKET")
    project_id = os.environ.get("GCP_PROJECT_ID")
    
    if bucket_name:
        st.success(f"Bucket: `{bucket_name}`")
    else:
        st.error("La variable GCP_STORAGE_BUCKET no está configurada.")

    if project_id:
        st.success(f"Proyecto: `{project_id}`")
    else:
        st.error("La variable GCP_PROJECT_ID no está configurada.")

    # 2. Verificar Acceso a Archivos en el Bucket
    st.subheader("2. Acceso a Archivos de Prompts")
    if bucket_name:
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            files_to_check = [
                "analisis-central.txt",
                "sintesis-que-evalua.txt",
                "recomendaciones.txt"
            ]
            
            all_files_ok = True
            for file in files_to_check:
                blob = bucket.blob(file)
                if blob.exists():
                    st.success(f"✅ {file} - Encontrado.")
                else:
                    st.error(f"❌ {file} - NO Encontrado.")
                    all_files_ok = False
            
            if not all_files_ok:
                 st.warning("Al menos un archivo no fue encontrado. Revisa los nombres y que estén en la raíz del bucket.")

        except Exception as e:
            st.error("🛑 Error al intentar conectar con el bucket o listar archivos.")
            st.code(f"Error detallado: {e}")
    else:
        st.warning("No se puede verificar el acceso a archivos porque la variable del bucket no está configurada.")

# --- FIN DEL PANEL DE DIAGNÓSTICO ---

# Inicializa Vertex AI una sola vez
if 'vertex_initialized' not in st.session_state:
    try:
        # Asegurarse que las variables de entorno no estén vacías
        if not GCP_PROJECT_ID or not GCP_LOCATION:
            st.sidebar.error("Las variables de entorno GCP_PROJECT_ID o GCP_LOCATION no están configuradas.")
            st.session_state.vertex_initialized = False
        else:
            vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            st.session_state.vertex_initialized = True
            st.sidebar.success("Vertex AI inicializado con éxito.")
    except Exception as e:
        st.session_state.vertex_initialized = False
        st.sidebar.error(f"Error al inicializar Vertex AI: {e}")

# --- PASO 1: Carga de Archivos ---
st.header("Paso 1: Carga tus Archivos")
col1, col2 = st.columns(2)
with col1:
    archivo_excel = st.file_uploader("Sube tu Excel con los datos base", type=["xlsx"])
with col2:
    archivo_plantilla = st.file_uploader("Sube tu Plantilla de Word", type=["docx"])

# --- PASO 2: Enriquecimiento con IA ---
st.header("Paso 2: Enriquece tus Datos con IA")
if st.button("🤖 Iniciar Análisis y Generación", disabled=(not st.session_state.vertex_initialized or not archivo_excel)):
    if not archivo_excel:
        st.warning("Por favor, sube un archivo Excel para continuar.")
    else:
        with st.spinner("Cargando prompts desde los archivos en Google Cloud Storage..."):
            st.session_state.prompts_cache['analisis'] = leer_prompt_desde_gcs("analisis-central.txt")
            st.session_state.prompts_cache['sintesis'] = leer_prompt_desde_gcs("sintesis-que-evalua.txt")
            st.session_state.prompts_cache['recomendaciones'] = leer_prompt_desde_gcs("recomendaciones.txt")

        if not all(st.session_state.prompts_cache.values()):
            st.error("No se pudieron cargar todos los prompts desde los archivos .txt en el bucket. Verifica que los archivos existan y los nombres sean correctos.")
        else:
            st.success("¡Prompts cargados con éxito desde los archivos!")
            
            # Creamos las instancias de los modelos de Gemini
            model_pro = GenerativeModel("gemini-2.0-flash")
            model_flash = GenerativeModel("gemini-2.0-flash-lite")

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
                progress_bar_main.progress((i + 1) / total_filas, text=f"Procesando ítem {i+1}/{total_filas}")

                with st.container(border=True):
                    try:
                        # --- LLAMADA 1: ANÁLISIS CENTRAL (RUTA COGNITIVA Y DISTRACTORES) ---
                        st.write(f"**Paso 1/3:** Realizando análisis central del ítem...")
                        prompt_paso1 = construir_prompt_paso1_analisis_central(fila, st.session_state.prompts_cache['analisis'])
                        response_paso1 = model_pro.generate_content(prompt_paso1)
                        analisis_central = response_paso1.text.strip()
                        time.sleep(1) 

                        header_correcta = "Ruta Cognitiva Correcta:"
                        header_distractores = "Análisis de Opciones No Válidas:"
                        idx_distractores = analisis_central.find(header_distractores)
                        
                        if idx_distractores == -1:
                            raise ValueError("No se encontró el separador 'Análisis de Opciones No Válidas' en la respuesta del paso 1.")

                        ruta_cognitiva = analisis_central[len(header_correcta):idx_distractores].strip()
                        analisis_distractores = analisis_central[idx_distractores:].strip()

                        # --- LLAMADA 2: SÍNTESIS DEL "QUÉ EVALÚA" ---
                        st.write(f"**Paso 2/3:** Sintetizando 'Qué Evalúa'...")
                        prompt_paso2 = construir_prompt_paso2_sintesis_que_evalua(analisis_central, fila, st.session_state.prompts_cache['sintesis'])
                        response_paso2 = model_flash.generate_content(prompt_paso2)
                        que_evalua = response_paso2.text.strip()
                        time.sleep(1)
                        
                        # --- LLAMADA 3: GENERACIÓN DE RECOMENDACIONES ---
                        st.write(f"**Paso 3/3:** Generando recomendaciones pedagógicas...")
                        prompt_paso3 = construir_prompt_paso3_recomendaciones(que_evalua, analisis_central, fila, st.session_state.prompts_cache['recomendaciones'])
                        response_paso3 = model_flash.generate_content(prompt_paso3)
                        recomendaciones = response_paso3.text.strip()
                        
                        titulo_avanzar = "RECOMENDACIÓN PARA AVANZAR"
                        idx_avanzar = recomendaciones.upper().find(titulo_avanzar)
                        
                        if idx_avanzar == -1:
                            raise ValueError("No se encontró el separador 'RECOMENDACIÓN PARA AVANZAR' en la respuesta del paso 3.")

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
                        st.error(f"Ocurrió un error procesando el ítem {item_id}: {e}")
                        df.loc[i, "Que_Evalua"] = "ERROR EN PROCESAMIENTO"
                        df.loc[i, "Justificacion_Correcta"] = f"Error: {e}"
                        df.loc[i, "Analisis_Distractores"] = "ERROR EN PROCESAMIENTO"
                        df.loc[i, "Recomendacion_Fortalecer"] = "ERROR EN PROCESAMIENTO"
                        df.loc[i, "Recomendacion_Avanzar"] = "ERROR EN PROCESAMIENTO"
            
            progress_bar_main.progress(1.0, text="¡Proceso completado!")
            st.session_state.df_enriquecido = df
            st.balloons()

# --- PASO 3: Subida a Cloud Storage y Verificación ---
if st.session_state.df_enriquecido is not None:
    st.header("Paso 3: Subida a la nube y verificación")
    st.dataframe(st.session_state.df_enriquecido.head())
    
    # Subida del Excel
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.df_enriquecido.to_excel(writer, index=False, sheet_name='Datos Enriquecidos')
    subir_a_cloud_storage(output_excel, "excel_enriquecido_con_ia.xlsx", 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- PASO 4: Ensamblaje de Fichas ---
if st.session_state.df_enriquecido is not None and archivo_plantilla is not None:
    st.header("Paso 4: Ensambla las Fichas Técnicas")
    
    columna_nombre_archivo = st.text_input(
        "Escribe el nombre de la columna para nombrar los archivos (ej. ItemId)",
        value="ItemId"
    )
    
    if st.button("📄 Ensamblar Fichas Técnicas y Subir a la Nube", type="primary"):
        df_final = st.session_state.df_enriquecido
        if columna_nombre_archivo not in df_final.columns:
            st.error(f"La columna '{columna_nombre_archivo}' no existe en el Excel. Por favor, elige una de: {', '.join(df_final.columns)}")
        else:
            with st.spinner("Ensamblando todas las fichas en un archivo .zip y subiendo a la Nube..."):
                plantilla_bytes = BytesIO(archivo_plantilla.getvalue())
                zip_buffer = BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    total_docs = len(df_final)
                    progress_bar_zip = st.progress(0, text="Iniciando ensamblaje...")
                    for i, fila in df_final.iterrows():
                        # Para cada fila, volvemos a cargar la plantilla desde los bytes originales
                        plantilla_bytes.seek(0)
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
                
                # Subir el ZIP a Cloud Storage
                subir_a_cloud_storage(zip_buffer, "fichas_tecnicas_generadas.zip", 'application/zip')
                st.session_state.zip_buffer = zip_buffer
                st.success("¡Ensamblaje y subida completados!")
