# Portafolio Cuantitativo y Dashboard MSCI COLCAP - Machine Learning & CRISP-ML

Este repositorio contiene la plataforma interactiva y el pipeline analítico del trabajo de grado de Maestría en Ciencia de Datos:
**"Predicción de la Dirección de la Tendencia de Precios en Acciones del Índice MSCI COLCAP mediante el Uso de Modelos de Aprendizaje de Máquina Supervisados e Indicadores de Análisis Técnico"**

- **Estudiante**: Vladimir Castillo Pérez
- **Director**: Juan Manuel González Álvarez
- **Institución**: Pontificia Universidad Javeriana Cali - Facultad de Ingeniería y Ciencias

---

## 🚀 Cómo Ejecutar el Dashboard

### Opción 1: En la Nube (Streamlit Community Cloud)
Puedes desplegar esta aplicación directamente y de forma gratuita conectando este repositorio de GitHub con [Streamlit Community Cloud](https://share.streamlit.io):
1. Inicia sesión con tu cuenta de GitHub en [share.streamlit.io](https://share.streamlit.io).
2. Haz clic en **"New app"**.
3. Selecciona este repositorio, rama `main` y en *Main file path* indica: `dashboard.py`.
4. Haz clic en **"Deploy"**.

### Opción 2: En GitHub Codespaces (Directo en el navegador)
1. En la página principal de este repositorio en GitHub, presiona el botón verde **`<> Code`**.
2. Ve a la pestaña **`Codespaces`** y selecciona **`Create codespace on main`**.
3. En la terminal integrada que se abrirá en tu navegador, ejecuta:
   ```bash
   pip install -r requirements.txt
   streamlit run dashboard.py
   ```
4. GitHub abrirá automáticamente una pestaña en tu navegador con el Dashboard ejecutándose en la nube.

### Opción 3: Localmente en tu Computadora
1. Clona el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
   cd TU_REPOSITORIO
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run dashboard.py
   ```
4. Abre tu navegador en: `http://localhost:8501/`

---

## 📊 Estructura del Proyecto

- `dashboard.py`: Aplicación principal de visualización interactiva, análisis exploratorio (EDA), auditoría PCA, entrenamiento de modelos, matrices de confusión, curvas ROC, simulación de backtesting y gobernanza bajo CRISP-ML.
- `requirements.txt`: Lista de dependencias y librerías requeridas.
- `acciones.xlsx` / `acciones_real.xlsx`: Datasets financieros de cotizaciones de la Bolsa de Valores de Colombia (bvc).
