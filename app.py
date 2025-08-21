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
from vertexai.preview.generative_models import GenerativeModel

# --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
st.set_page_config(
    page_title="Ensamblador de Fichas Técnicas Google Vertex AI",
    page_icon="🤖",
    layout="wide"
)

# --- VARIABLES DE ENTORNO ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION")
GCP_STORAGE_BUCKET = os.environ.get("GCP_STORAGE_BUCKET")

# --- FUNCIONES DE LÓGICA (sin cambios) ---
def limpiar_html(texto_html):
    if not isinstance(texto_html, str): return texto_html
    return re.sub(re.compile('<.*?>'), '', texto_html)

def leer_prompt_desde_gcs(nombre_archivo):
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
        return blob.download_as_text()
    except Exception as e:
        st.error(f"Error al LEER el archivo '{nombre_archivo}'. Causa raíz: {e}")
        return None

def subir_a_cloud_storage(data_buffer, file_name, content_type):
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
    fila = fila.fillna('')
    return prompt_template.format(
        ItemContexto=fila.get('ItemContexto', 'No aplica'), ItemEnunciado=fila.get('ItemEnunciado', 'No aplica'),
        ComponenteNombre=fila.get('ComponenteNombre', 'No aplica'), CompetenciaNombre=fila.get('CompetenciaNombre', ''),
        AfirmacionNombre=fila.get('AfirmacionNombre', ''), EvidenciaNombre=fila.get('EvidenciaNombre', ''),
        Tipologia_Textual=fila.get('Tipologia Textual', 'No aplica'), ItemGradoId=fila.get('ItemGradoId', ''),
        Analisis_Errores=fila.get('Analisis_Errores', 'No aplica'), AlternativaClave=fila.get('AlternativaClave', 'No aplica'),
        OpcionA=fila.get('OpcionA', 'No aplica'), OpcionB=fila.get('OpcionB', 'No aplica'),
        OpcionC=fila.get('OpcionC', 'No aplica'), OpcionD=fila.get('OpcionD', 'No aplica')
    )

def construir_prompt_paso2_sintesis_que_evalua(analisis_central_generado, fila, prompt_template):
    fila = fila.fillna('')
    try:
        header_distractores = "Análisis de Opciones No Válidas:"
        idx_distractores = analisis_central_generado.find(header_distractores)
        ruta_cognitiva_texto = analisis_central_generado[:idx_distractores].strip() if idx_distractores != -1 else analisis_central_generado
    except:
        ruta_cognitiva_texto = analisis_central_generado
    return prompt_template.format(
        ruta_cognitiva_texto=ruta_cognitiva_texto, CompetenciaNombre=fila.get('CompetenciaNombre', ''),
        AfirmacionNombre=fila.get('AfirmacionNombre', ''), EvidenciaNombre=fila.get('EvidenciaNombre', '')
    )

def construir_prompt_paso3_recomendaciones(que_evalua_sintetizado, analisis_central_generado, fila, prompt_template):
    fila = fila.fillna('')
    return prompt_template.format(
        que_evalua_sintetizado=que_evalua_sintetizado, analisis_central_generado=analisis_central_generado,
        ItemContexto=fila.get('ItemContexto', 'No aplica'), ItemEnunciado=fila.get('ItemEnunciado', 'No aplica'),
        ComponenteNombre=fila.get('ComponenteNombre', 'No aplica'), CompetenciaNombre=fila.get('CompetenciaNombre', ''),
        AfirmacionNombre=fila.get('AfirmacionNombre', ''), EvidenciaNombre=fila.get('EvidenciaNombre', ''),
        Tipologia_Textual=fila.get('Tipologia Textual', 'No aplica'), ItemGradoId=fila.get('ItemGradoId', ''),
        Analisis_Errores=fila.get('Analisis_Errores', 'No aplica'), AlternativaClave=fila.get('AlternativaClave', 'No aplica')
    )

# --- INTERFAZ PRINCIPAL DE STREAMLIT ---
st.title("🤖 Ensamblador de Fichas Técnicas con Google Vertex IA")
st.markdown("Una aplicación para enriquecer datos pedagógicos y generar fichas personalizadas.")

if 'df_enriquecido' not in st.session_state: st.session_state.df_enriquecido = None
if 'zip_buffer' not in st.session_state: st.session_state.zip_buffer = None
if 'prompts_cache' not in st.session_state: st.session_state.prompts_cache = {}

# --- PASO 0: Configuración y Validación ---
st.sidebar.header("🔑 Configuración")
st.info("Esta aplicación usa Google Cloud Storage para leer los prompts y guardar los resultados.")

with st.sidebar.expander("🔍 Panel de Diagnóstico de Sistema", expanded=True):
    st.write("Verificando la configuración y el acceso a los prompts...")
    st.subheader("1. Variables de Entorno")
    bucket_name = os.environ.get("GCP_STORAGE_BUCKET")
    project_id = os.environ.get("GCP_PROJECT_ID")
    if bucket_name: st.success(f"Bucket: `{bucket_name}`")
    else: st.error("La variable GCP_STORAGE_BUCKET no está configurada.")
    if project_id: st.success(f"Proyecto: `{project_id}`")
    else: st.error("La variable GCP_PROJECT_ID no está configurada.")

    st.subheader("2. Acceso a Archivos de Prompts")
    if bucket_name:
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            # --- MODIFICACIÓN: Añadir el nuevo prompt a la verificación ---
            files_to_check = ["analisis-central.txt", "sintesis-que-evalua.txt", "recomendaciones.txt", "parafraseo.txt"]
            all_files_ok = True
            for file in files_to_check:
                blob = bucket.blob(file)
                if blob.exists(): st.success(f"✅ {file} - Encontrado.")
                else:
                    st.error(f"❌ {file} - NO Encontrado.")
                    all_files_ok = False
            if not all_files_ok:
                st.warning("Al menos un archivo no fue encontrado. Revisa los nombres y que estén en la raíz del bucket.")
        except Exception as e:
            st.error(f"🛑 Error al intentar conectar con el bucket o listar archivos: {e}")
    else:
        st.warning("No se puede verificar el acceso a archivos porque la variable del bucket no está configurada.")

if 'vertex_initialized' not in st.session_state:
    try:
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
with col1: archivo_excel = st.file_uploader("Sube tu Excel con los datos base", type=["xlsx"])
with col2: archivo_plantilla = st.file_uploader("Sube tu Plantilla de Word", type=["docx"])

# --- PASO 2: Enriquecimiento con IA ---
st.header("Paso 2: Enriquece tus Datos con IA")
if st.button("🤖 Iniciar Análisis y Generación", disabled=(not st.session_state.vertex_initialized or not archivo_excel)):
    if not archivo_excel:
        st.warning("Por favor, sube un archivo Excel para continuar.")
    else:
        with st.spinner("Cargando prompts desde Google Cloud Storage..."):
            st.session_state.prompts_cache['analisis'] = leer_prompt_desde_gcs("analisis-central.txt")
            st.session_state.prompts_cache['sintesis'] = leer_prompt_desde_gcs("sintesis-que-evalua.txt")
            st.session_state.prompts_cache['recomendaciones'] = leer_prompt_desde_gcs("recomendaciones.txt")
            # --- MODIFICACIÓN: Cargar el nuevo prompt de parafraseo ---
            st.session_state.prompts_cache['parafraseo'] = leer_prompt_desde_gcs("parafraseo.txt")

        if not all(st.session_state.prompts_cache.values()):
            st.error("No se pudieron cargar todos los prompts. Verifica que los archivos existan y los nombres sean correctos.")
        else:
            st.success("¡Prompts cargados con éxito!")
            
            model_pro = GenerativeModel("gemini-1.5-pro-001")
            model_flash = GenerativeModel("gemini-1.5-flash-001")

            with st.spinner("Procesando archivo Excel y preparando datos..."):
                df = pd.read_excel(archivo_excel)
                for col in df.columns:
                    if df[col].dtype == 'object': df[col] = df[col].apply(limpiar_html)
                
                columnas_nuevas = ["Que_Evalua", "Justificacion_Correcta", "Analisis_Distractores",
                                   "Justificacion_A", "Justificacion_B", "Justificacion_C", "Justificacion_D",
                                   "Recomendacion_Fortalecer", "Recomendacion_Avanzar", "oportunidad_de_mejora"]
                for col in columnas_nuevas:
                    if col not in df.columns: df[col] = ""
                st.success("Datos limpios y listos.")

            progress_bar_main = st.progress(0, text="Iniciando Proceso...")
            total_filas = len(df)

            for i, fila in df.iterrows():
                item_id = fila.get('ItemId', i + 1)
                st.markdown(f"--- \n ### Procesando Ítem: **{item_id}**")
                progress_bar_main.progress((i + 1) / total_filas, text=f"Procesando ítem {i+1}/{total_filas}")

                with st.container(border=True):
                    try:
                        # --- PASO 1: ANÁLISIS CENTRAL ---
                        st.write(f"**Paso 1/4:** Realizando análisis central...")
                        prompt_paso1 = construir_prompt_paso1_analisis_central(fila, st.session_state.prompts_cache['analisis'])
                        response_paso1 = model_pro.generate_content(prompt_paso1)
                        analisis_central = response_paso1.text.strip()
                        time.sleep(1)

                        # --- MODIFICACIÓN: Lógica para separar Ruta Cognitiva y Distractores ---
                        header_correcta = "Ruta Cognitiva Correcta:"
                        header_distractores = "Análisis de Opciones No Válidas:"
                        idx_distractores = analisis_central.find(header_distractores)

                        if idx_distractores != -1:
                            ruta_cognitiva = analisis_central[len(header_correcta):idx_distractores].strip()
                            analisis_distractores_bloque = analisis_central[idx_distractores + len(header_distractores):].strip()
                            df.loc[i, "Justificacion_Correcta"] = ruta_cognitiva
                            df.loc[i, "Analisis_Distractores"] = analisis_distractores_bloque

                            # Separar cada justificación de distractor en su propia celda
                            clave_correcta = str(fila.get('AlternativaClave', '')).strip().upper()
                            opciones = ['A', 'B', 'C', 'D']
                            for opt in opciones:
                                if opt != clave_correcta:
                                    pattern = re.compile(rf"Opción\s*{opt}:\s*(.*?)(?=\s*Opción\s*[A-D]:|$)", re.DOTALL | re.IGNORECASE)
                                    match = pattern.search(analisis_distractores_bloque)
                                    if match:
                                        df.loc[i, f"Justificacion_{opt}"] = match.group(1).strip()
                        else:
                            df.loc[i, "Justificacion_Correcta"] = analisis_central # Si falla el parseo, guarda todo
                            df.loc[i, "Analisis_Distractores"] = "Error al parsear distractores"

                        # --- PASO 2: SÍNTESIS DEL "QUÉ EVALÚA" ---
                        st.write(f"**Paso 2/4:** Sintetizando 'Qué Evalúa'...")
                        prompt_paso2 = construir_prompt_paso2_sintesis_que_evalua(analisis_central, fila, st.session_state.prompts_cache['sintesis'])
                        response_paso2 = model_flash.generate_content(prompt_paso2)
                        que_evalua = response_paso2.text.strip()
                        df.loc[i, "Que_Evalua"] = que_evalua
                        time.sleep(1)
                        
                        # --- PASO 3: GENERACIÓN DE RECOMENDACIONES ---
                        st.write(f"**Paso 3/4:** Generando recomendaciones...")
                        prompt_paso3 = construir_prompt_paso3_recomendaciones(que_evalua, analisis_central, fila, st.session_state.prompts_cache['recomendaciones'])
                        response_paso3 = model_flash.generate_content(prompt_paso3)
                        recomendaciones = response_paso3.text.strip()
                        
                        idx_avanzar = recomendaciones.upper().find("RECOMENDACIÓN PARA AVANZAR")
                        if idx_avanzar != -1:
                            fortalecer = recomendaciones[:idx_avanzar].replace("RECOMENDACIÓN PARA FORTALECER", "").strip()
                            avanzar = recomendaciones[idx_avanzar:].replace("RECOMENDACIÓN PARA AVANZAR", "").strip()
                        else:
                            fortalecer = recomendaciones.replace("RECOMENDACIÓN PARA FORTALECER", "").strip()
                            avanzar = "No generada"
                        
                        df.loc[i, "Recomendacion_Fortalecer"] = fortalecer
                        df.loc[i, "Recomendacion_Avanzar"] = avanzar

                        # --- MODIFICACIÓN: PASO 4: PARAFRASEO PARA OPORTUNIDAD DE MEJORA ---
                        if fortalecer != "No generada" and fortalecer.strip() != "":
                            st.write(f"**Paso 4/4:** Creando oportunidad de mejora...")
                            prompt_parafraseo = st.session_state.prompts_cache['parafraseo'].format(recomendacion_fortalecer=fortalecer)
                            response_parafraseo = model_flash.generate_content(prompt_parafraseo)
                            oportunidad = response_parafraseo.text.strip()
                            df.loc[i, "oportunidad_de_mejora"] = oportunidad
                        else:
                            df.loc[i, "oportunidad_de_mejora"] = "No se generó recomendación para fortalecer."

                        st.success(f"Ítem {item_id} procesado con éxito.")

                    except Exception as e:
                        st.error(f"Ocurrió un error procesando el ítem {item_id}: {e}")
                        for col in columnas_nuevas: df.loc[i, col] = f"ERROR: {e}"
            
            progress_bar_main.progress(1.0, text="¡Proceso completado!")
            st.session_state.df_enriquecido = df
            st.balloons()
            
# --- PASO 3: Subida a Cloud Storage y Verificación ---
if st.session_state.df_enriquecido is not None:
    st.header("Paso 3: Subida a la nube y verificación")
    st.dataframe(st.session_state.df_enriquecido.head())
    output_excel = BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.df_enriquecido.to_excel(writer, index=False, sheet_name='Datos Enriquecidos')
    subir_a_cloud_storage(output_excel, "excel_enriquecido_con_ia.xlsx", 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- PASO 4: Ensamblaje de Fichas ---
if st.session_state.df_enriquecido is not None and archivo_plantilla is not None:
    st.header("Paso 4: Ensambla las Fichas Técnicas")
    columna_nombre_archivo = st.text_input("Escribe el nombre de la columna para nombrar los archivos (ej. ItemId)", value="ItemId")
    if st.button("📄 Ensamblar Fichas Técnicas y Subir a la Nube", type="primary"):
        df_final = st.session_state.df_enriquecido
        if columna_nombre_archivo not in df_final.columns:
            st.error(f"La columna '{columna_nombre_archivo}' no existe en el Excel. Elige una de: {', '.join(df_final.columns)}")
        else:
            with st.spinner("Ensamblando todas las fichas en un archivo .zip y subiendo a la Nube..."):
                plantilla_bytes = BytesIO(archivo_plantilla.getvalue())
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    total_docs = len(df_final)
                    progress_bar_zip = st.progress(0, text="Iniciando ensamblaje...")
                    for i, fila in df_final.iterrows():
                        plantilla_bytes.seek(0)
                        doc = DocxTemplate(plantilla_bytes)
                        contexto = fila.to_dict()
                        contexto_limpio = {k: (v if pd.notna(v) else "") for k, v in contexto.items()}
                        doc.render(contexto_limpio)
                        doc_buffer = BytesIO()
                        doc.save(doc_buffer)
                        nombre_base = str(fila.get(columna_nombre_archivo, f"ficha_{i+1}")).replace('/', '_').replace('\\', '_')
                        zip_file.writestr(f"{nombre_base}.docx", doc_buffer.getvalue())
                        progress_bar_zip.progress((i + 1) / total_docs, text=f"Añadiendo ficha {i+1}/{total_docs} al .zip")
                subir_a_cloud_storage(zip_buffer, "fichas_tecnicas_generadas.zip", 'application/zip')
                st.session_state.zip_buffer = zip_buffer
                st.success("¡Ensamblaje y subida completados!")
