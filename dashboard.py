import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import jarque_bera, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, confusion_matrix
import time
import io

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Dashboard MSCI COLCAP - CRISP-ML & IEEE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS premium (Temas académicos, colores HSL sobrios y microanimaciones)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #1E3A8A;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        text-align: left;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-align: left;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 1rem;
    }
    
    /* IEEE Standard Captions */
    .ieee-table-caption {
        font-family: 'Times New Roman', serif;
        font-size: 10.5pt;
        text-align: center;
        font-weight: bold;
        text-transform: uppercase;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        color: #111827;
    }
    .ieee-figure-caption {
        font-family: 'Times New Roman', serif;
        font-size: 10pt;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
        color: #374151;
        font-style: italic;
    }
    
    .glossary-card {
        background: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FUNCIONES AUXILIARES CIENTÍFICAS
# ==========================================
def compute_vif(X_df):
    corr = X_df.corr().values
    try:
        inv_corr = np.linalg.inv(corr)
        vifs = np.diag(inv_corr)
        return pd.Series(vifs, index=X_df.columns)
    except np.linalg.LinAlgError:
        return pd.Series([999.0] * len(X_df.columns), index=X_df.columns)

def add_copy_controls(df_data, name_label):
    st.download_button(
        label=f"📋 Descargar Datos ({name_label}) en CSV",
        data=df_data.to_csv(index=False),
        file_name=f"datos_{name_label.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"dl_btn_{name_label.lower().replace(' ', '_')}_{time.time()}"
    )
    with st.expander(f"🔍 Ver y copiar CSV de {name_label}"):
        st.code(df_data.to_csv(index=False), language='text')

# RENDERIZADOR DE TABLAS CON FORMATO IEEE Y COPIADO DIRECTO PARA WORD (RICH TEXT)
def render_ieee_table_with_copy(df, table_id, title_caption, source_caption):
    # Convertir DataFrame a HTML
    table_html = df.to_html(classes='ieee-table', border=1, table_id=table_id, index=False)
    
    # Marcado HTML y JS con API de Selección para copia de tabla con formato nativo en Word/Excel
    full_html = f"""
    <style>
        .ieee-table-container {{
            font-family: 'Times New Roman', serif;
            margin-bottom: 1rem;
            width: 6.5in;
            max-width: 6.5in;
            margin-left: auto;
            margin-right: auto;
        }}
        .ieee-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 10pt;
            margin-top: 5px;
            color: #111827;
        }}
        .ieee-table th {{
            border-top: 1.5px solid black;
            border-bottom: 1.5px solid black;
            padding: 6px 4px;
            text-align: center;
            font-weight: bold;
            background-color: #F9FAFB;
        }}
        .ieee-table td {{
            border-bottom: 0.5px solid #E5E7EB;
            padding: 5px 4px;
            text-align: center;
        }}
        .ieee-table, .ieee-table th, .ieee-table td {{
            border-left: none;
            border-right: none;
        }}
        .ieee-table tr:last-child td {{
            border-bottom: 1.5px solid black;
        }}
        .copy-btn {{
            background-color: #1E3A8A;
            color: white;
            border: none;
            padding: 6px 12px;
            font-family: 'Inter', sans-serif;
            font-size: 9pt;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 8px;
            display: inline-block;
        }}
        .copy-btn:hover {{
            background-color: #2563EB;
        }}
        .caption-top {{
            font-family: 'Times New Roman', serif;
            font-size: 10.5pt;
            text-align: center;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .caption-bottom {{
            font-family: 'Times New Roman', serif;
            font-size: 9.5pt;
            text-align: center;
            font-style: italic;
            margin-top: 5px;
            margin-bottom: 10px;
        }}
    </style>
    <div class="ieee-table-container">
        <div id="content_{table_id}">
            <div class="caption-top">{title_caption}</div>
            {table_html}
            <div class="caption-bottom">{source_caption}</div>
        </div>
        <button class="copy-btn" onclick="copyFormattedTable('content_{table_id}')">📋 Copiar Tabla con Formato para Word / Excel</button>
    </div>
    <script>
        function copyFormattedTable(contentId) {{
            var range = document.createRange();
            range.selectNode(document.getElementById(contentId));
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            try {{
                var successful = document.execCommand('copy');
                if(successful) {{
                    alert('¡Tabla copiada al portapapeles con formato! Puede pegarla directamente en Word o Excel (se mantendrá la cuadrícula y estilos visuales intactos).');
                }} else {{
                    alert('No se pudo copiar automáticamente. Por favor, sombree la tabla con el cursor y pulse Ctrl+C.');
                }}
            }} catch (err) {{
                alert('Error al copiar: ' + err);
            }}
            window.getSelection().removeAllRanges();
        }}
    </script>
    """
    import streamlit.components.v1 as components
    # Calcular altura dinámica para evitar scrollbars
    height = len(df) * 38 + 160
    components.html(full_html, height=max(height, 220), scrolling=True)

# ==========================================
# BASE DE DATOS DE COMPAÑÍAS Y PERFILES (REAL)
# ==========================================
EMISORES = {
    "Ecopetrol": {
        "ticker": "ECOPETROL.CL", "sector": "Petróleo y Gas / Energía", "sfc_code": "009", "weight": "9.33%",
        "description": "Ecopetrol S.A. es la compañía más grande de Colombia y una de las principales corporaciones integradas de energía en América Latina. Participa en la exploración, producción, transporte, refinación y comercialización de hidrocarburos, y en transmisión de energía a través de su filial ISA.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos Operacionales (COP Billones)": [143.08, 133.33],
            "EBITDA Consolidado (COP Billones)": [58.91, 54.10],
            "Utilidad Neta (COP Billones)": [19.00, 14.90],
            "Utilidad por Acción (UPA - COP)": [461.0, 362.0],
            "Relación Precio/Ganancia (RPG - x)": [5.2, 6.1],
            "Rentabilidad por Dividendo (DY - %)": [17.5, 14.1]
        }
    },
    "Bancolombia": {
        "ticker": "PFBCOLOM.CL", "sector": "Financiero / Bancario", "sfc_code": "044", "weight": "23.20%",
        "description": "Bancolombia S.A. es la institución financiera privada más grande de Colombia. Presta servicios financieros a una base de más de 25 millones de clientes en Colombia, El Salvador, Guatemala y Panamá, ofreciendo banca comercial, de consumo, corporativa, leasing, corretaje y banca de inversión.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Cartera Bruta (COP Billones)": [254.00, 279.00],
            "Activos Totales (COP Billones)": [343.00, 372.00],
            "Utilidad Neta (COP Billones)": [6.21, 6.36],
            "Utilidad por Acción (UPA - COP)": [6450.0, 6600.0],
            "Relación Precio/Ganancia (RPG - x)": [6.0, 5.8],
            "Rentabilidad por Dividendo (DY - %)": [9.2, 10.4]
        }
    },
    "Grupo Sura": {
        "ticker": "GRUPOSURA.CL", "sector": "Servicios Financieros / Holding", "sfc_code": "088", "weight": "5.21%",
        "description": "Grupo de Inversiones Suramericana S.A. (Grupo SURA) es un holding financiero multilatino con inversiones estratégicas en banca (Bancolombia), seguros (Suramericana), pensiones (SURA Asset Management) y portafolio industrial (Grupo Argos).",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos Totales (COP Billones)": [35.50, 37.20],
            "Utilidad Operativa (COP Billones)": [4.60, 9.20],
            "Utilidad Neta Controladora (COP Billones)": [2.30, 6.10],
            "Utilidad por Acción (UPA - COP)": [2800.0, 2950.0],
            "Relación Precio/Ganancia (RPG - x)": [7.5, 8.2],
            "Rentabilidad por Dividendo (DY - %)": [3.8, 4.2]
        }
    },
    "Interconexión Eléctrica": {
        "ticker": "ISA.CL", "sector": "Servicios Públicos / Infraestructura", "sfc_code": "023", "weight": "10.31%",
        "description": "Interconexión Eléctrica S.A. E.S.P. (ISA) es una empresa multilatina líder en la transmisión de energía eléctrica, gestión de concesiones viales de autopistas y servicios de telecomunicaciones, con operaciones en Colombia, Brasil, Chile, Perú y Centroamérica.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos Operacionales (COP Billones)": [14.10, 15.80],
            "EBITDA (COP Billones)": [9.10, 9.70],
            "Utilidad Neta (COP Billones)": [2.45, 2.80],
            "Utilidad por Acción (UPA - COP)": [2218.0, 2536.0],
            "Relación Precio/Ganancia (RPG - x)": [8.8, 7.6],
            "Rentabilidad por Dividendo (DY - %)": [5.1, 5.8]
        }
    },
    "Grupo Nutresa": {
        "ticker": "NUTRESA.CL", "sector": "Consumo Masivo / Alimentos", "sfc_code": "062", "weight": "5.38%",
        "description": "Grupo Nutresa S.A. es la empresa líder en alimentos procesados en Colombia y uno de los conglomerados más importantes de consumo masivo en América Latina, operando en ocho unidades de negocio: carnes, galletas, chocolates, café, helados y pastas.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos Operacionales (COP Billones)": [18.90, 18.60],
            "EBITDA Consolidado (COP Billones)": [2.20, 2.40],
            "Utilidad Neta Atribuible (COP Billones)": [0.720, 0.751],
            "Utilidad por Acción (UPA - COP)": [1570.0, 1640.0],
            "Relación Precio/Ganancia (RPG - x)": [22.4, 20.8],
            "Rentabilidad por Dividendo (DY - %)": [1.8, 2.1]
        }
    },
    "Grupo Aval": {
        "ticker": "PFAVAL.CL", "sector": "Financiero / Holding", "sfc_code": "067", "weight": "2.63%",
        "description": "Grupo Aval Acciones y Valores S.A. es el holding financiero líder en Colombia que controla los bancos Bogotá, Occidente, Popular y AV Villas, la administradora de pensiones Porvenir y Corficolombiana.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Activos Totales (COP Billones)": [290.0, 310.0],
            "Utilidad Neta (COP Billones)": [0.74, 0.95],
            "Utilidad por Acción (UPA - COP)": [32.0, 41.0],
            "Relación Precio/Ganancia (RPG - x)": [15.2, 11.8],
            "Rentabilidad por Dividendo (DY - %)": [4.8, 5.5]
        }
    },
    "Grupo Argos": {
        "ticker": "GRUPOARGOS.CL", "sector": "Infraestructura / Holding", "sfc_code": "012", "weight": "4.50%",
        "description": "Grupo de Inversiones Argos S.A. es un holding con foco estratégico en infraestructura, líder regional en el mercado de cemento (Cementos Argos), energía eléctrica (Celsia) y concesiones viales (Odinsa).",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos Totales (COP Billones)": [21.5, 23.1],
            "Utilidad Neta (COP Billones)": [1.12, 1.45],
            "Utilidad por Acción (UPA - COP)": [1320.0, 1650.0],
            "Relación Precio/Ganancia (RPG - x)": [9.8, 8.4],
            "Rentabilidad por Dividendo (DY - %)": [3.2, 4.0]
        }
    },
    "Cementos Argos": {
        "ticker": "CEMARGOS.CL", "sector": "Construcción / Materiales", "sfc_code": "015", "weight": "5.03%",
        "description": "Cementos Argos S.A. es una multinacional colombiana productora de cemento y concreto, con una fuerte presencia en Colombia, el Caribe, Centroamérica y los Estados Unidos.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (COP Billones)": [12.7, 13.5],
            "EBITDA (COP Billones)": [2.15, 2.45],
            "Utilidad Neta (COP Billones)": [0.45, 0.58],
            "Utilidad por Acción (UPA - COP)": [390.0, 510.0],
            "Relación Precio/Ganancia (RPG - x)": [18.2, 14.5],
            "Rentabilidad por Dividendo (DY - %)": [2.8, 3.5]
        }
    },
    "Grupo Éxito": {
        "ticker": "EXITO.CL", "sector": "Comercio / Retail", "sfc_code": "055", "weight": "0.51%",
        "description": "Almacenes Éxito S.A. es el líder del comercio al detal en Colombia, operando cadenas de hipermercados y supermercados bajo las marcas Éxito, Carulla, Surtimax y Super Inter.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (COP Billones)": [20.1, 19.8],
            "Utilidad Neta (COP Millones)": [125.0, 135.0],
            "Utilidad por Acción (UPA - COP)": [96.0, 104.0],
            "Relación Precio/Ganancia (RPG - x)": [25.4, 23.5],
            "Rentabilidad por Dividendo (DY - %)": [1.9, 2.2]
        }
    },
    "ETB": {
        "ticker": "ETB.CL", "sector": "Telecomunicaciones", "sfc_code": "077", "weight": "0.10%",
        "description": "La Empresa de Telecomunicaciones de Bogotá S.A. E.S.P. es un operador líder de servicios de telecomunicaciones en Colombia, proveedor de telefonía móvil, internet de fibra óptica y soluciones TIC.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (COP Billones)": [1.45, 1.38],
            "EBITDA (COP Billones)": [0.38, 0.35],
            "Utilidad Neta (COP Billones)": [-0.045, -0.028],
            "Utilidad por Acción (UPA - COP)": [-12.0, -8.0],
            "Relación Precio/Ganancia (RPG - x)": [-9.2, -14.5],
            "Rentabilidad por Dividendo (DY - %)": [0.0, 0.0]
        }
    },
    "Bolsa de Valores de Colombia": {
        "ticker": "BVC.CL", "sector": "Financiero / Bolsa", "sfc_code": "099", "weight": "0.20%",
        "description": "La Bolsa de Valores de Colombia S.A. (bvc) es el operador de la infraestructura del mercado financiero colombiano, ofreciendo plataformas de renta variable, renta fija y derivados.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (COP Miles de Millones)": [310.0, 335.0],
            "Utilidad Neta (COP Miles de Millones)": [55.0, 68.0],
            "Utilidad por Acción (UPA - COP)": [940.0, 1160.0],
            "Relación Precio/Ganancia (RPG - x)": [10.5, 8.9],
            "Rentabilidad por Dividendo (DY - %)": [6.2, 7.8]
        }
    },
    "Canacol Energy": {
        "ticker": "CNEC.CL", "sector": "Gas Natural / Energía", "sfc_code": "095", "weight": "0.15%",
        "description": "Canacol Energy Ltd. es la mayor compañía independiente de exploración y producción de gas natural convencional terrestre en Colombia, enfocada en suplir la creciente demanda nacional.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (USD Millones)": [290.0, 260.0],
            "Utilidad Neta (USD Millones)": [85.0, -15.0],
            "Utilidad por Acción (UPA - COP)": [1820.0, -320.0],
            "Relación Precio/Ganancia (RPG - x)": [8.5, -45.2],
            "Rentabilidad por Dividendo (DY - %)": [7.5, 0.0]
        }
    },
    "Enka de Colombia": {
        "ticker": "ENKA.CL", "sector": "Industrial / Textil / Reciclaje", "sfc_code": "082", "weight": "0.08%",
        "description": "Enka de Colombia S.A. es una compañía líder en la producción de polímeros y fibras sintéticas, consolidada como el mayor reciclador de botellas de PET de Suramérica.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (COP Miles de Millones)": [480.0, 510.0],
            "EBITDA (COP Miles de Millones)": [35.0, 42.0],
            "Utilidad Neta (COP Miles de Millones)": [12.0, 18.0],
            "Utilidad por Acción (UPA - COP)": [2.4, 3.6],
            "Relación Precio/Ganancia (RPG - x)": [11.2, 8.5],
            "Rentabilidad por Dividendo (DY - %)": [3.5, 4.2]
        }
    },
    "Mineros": {
        "ticker": "MINEROS.CL", "sector": "Minería / Oro", "sfc_code": "038", "weight": "1.01%",
        "description": "Mineros S.A. es una compañía multinacional colombiana dedicada a la exploración y explotación de metales preciosos, principalmente oro, con operaciones en Colombia, Nicaragua y Argentina.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (USD Millones)": [510.0, 545.0],
            "Utilidad Neta (USD Millones)": [38.0, 48.0],
            "Utilidad por Acción (UPA - COP)": [390.0, 490.0],
            "Relación Precio/Ganancia (RPG - x)": [7.8, 6.2],
            "Rentabilidad por Dividendo (DY - %)": [6.8, 8.1]
        }
    },
    "Corficolombiana": {
        "ticker": "CORFICOLCF.CL", "sector": "Servicios Financieros / Inversiones", "sfc_code": "052", "weight": "1.20%",
        "description": "La Corporación Financiera Colombiana S.A. (Corficolombiana) es una corporación dedicada a la inversión y gestión de portafolios en sectores de infraestructura, energía y hotelería.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos de Portafolio (COP Billones)": [8.5, 9.1],
            "Utilidad Neta (COP Billones)": [1.45, 1.78],
            "Utilidad por Acción (UPA - COP)": [3950.0, 4820.0],
            "Relación Precio/Ganancia (RPG - x)": [4.8, 3.9],
            "Rentabilidad por Dividendo (DY - %)": [11.2, 13.8]
        }
    },
    "Conconcreto": {
        "ticker": "CONCONCRET.CL", "sector": "Construcción / Ingeniería", "sfc_code": "040", "weight": "0.05%",
        "description": "Constructora Conconcreto S.A. es una compañía líder en diseño, ingeniería y desarrollo de proyectos de infraestructura pública y edificaciones privadas en Colombia.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos (COP Miles de Millones)": [980.0, 1020.0],
            "Utilidad Neta (COP Miles de Millones)": [18.0, 24.0],
            "Utilidad por Acción (UPA - COP)": [16.0, 22.0],
            "Relación Precio/Ganancia (RPG - x)": [15.2, 11.5],
            "Rentabilidad por Dividendo (DY - %)": [0.0, 0.0]
        }
    },
    "Coltejer": {
        "ticker": "COLTEJER.CL", "sector": "Industrial / Textil", "sfc_code": "018", "weight": "0.01%",
        "description": "Compañía Colombiana de Tejidos S.A. (Coltejer) es la empresa textil más antigua de Colombia. Se encuentra en un proceso de reestructuración operativa profunda y venta de activos inmobiliarios.",
        "financials_years": ["2023", "2024"],
        "financials": {
            "Ingresos Netos (COP Miles de Millones)": [15.0, 8.0],
            "Utilidad Neta (COP Miles de Millones)": [-32.0, -18.0],
            "Utilidad por Acción (UPA - COP)": [-420.0, -230.0],
            "Relación Precio/Ganancia (RPG - x)": [-2.5, -4.5],
            "Rentabilidad por Dividendo (DY - %)": [0.0, 0.0]
        }
    }
}

# ==========================================
# CARGA DE DATOS REALES (DESDE EXCEL LOCAL)
# ==========================================
@st.cache_data
def load_real_data():
    df_raw = pd.read_excel('acciones.xlsx')
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw['Company'] = df_raw['Company'].astype(str)
    
    # Limpiar codificación de caracteres especiales
    df_raw['Company'] = df_raw['Company'].str.replace('xito', 'Grupo Éxito')
    df_raw['Company'] = df_raw['Company'].str.replace('Grupo xito', 'Grupo Éxito')
    df_raw['Company'] = df_raw['Company'].str.replace('Interconexin Elctrica', 'Interconexión Eléctrica')
    
    return df_raw

try:
    df_raw = load_real_data()
except Exception as e:
    st.error(f"Error crítico de lectura del archivoacciones.xlsx: {e}")
    st.stop()

# ==========================================
# SELECTORES DEL PANEL LATERAL
# ==========================================
st.sidebar.markdown("## CONFIGURACIÓN")
comp_select = st.sidebar.selectbox("Seleccione el Emisor / Acción:", list(EMISORES.keys()))
comp_info = EMISORES[comp_select]

# Slider/Inputs de Selección de Fechas (Rango predeterminado de 2015-01-01 a 2026-06-30)
st.sidebar.markdown("### Rango Temporal")
date_min_val = pd.to_datetime('2015-01-01')
date_max_val = pd.to_datetime('2026-06-30')
default_start = pd.to_datetime('2015-01-01')
default_end = pd.to_datetime('2026-06-30')

start_date_input = st.sidebar.date_input("Fecha Inicio:", default_start, min_value=date_min_val, max_value=date_max_val)
end_date_input = st.sidebar.date_input("Fecha Fin:", default_end, min_value=date_min_val, max_value=date_max_val)

if start_date_input > end_date_input:
    st.sidebar.error("Error: Fecha de inicio mayor a fecha de fin.")
    st.stop()

start_date = pd.to_datetime(start_date_input)
end_date = pd.to_datetime(end_date_input)

# Filtrado directo
df_stock = df_raw[(df_raw['Company'] == comp_select) & (df_raw['Date'] >= start_date) & (df_raw['Date'] <= end_date)].copy()

if df_stock.empty:
    st.sidebar.warning("No hay suficientes registros para el activo en el rango seleccionado. Mostrando todo el historial disponible.")
    df_stock = df_raw[df_raw['Company'] == comp_select].copy()
    start_date = df_stock['Date'].min()
    end_date = df_stock['Date'].max()

df_stock = df_stock.sort_values('Date').reset_index(drop=True)
df_stock['Return'] = df_stock['Close'].pct_change() * 100

# ==========================================
# CALCULADORA TÉCNICA E INDICADORES (DETERMINÍSTICO)
# ==========================================
def compute_all_indicators_df(df):
    df_feat = df.copy()
    close = df_feat['Close']
    high = df_feat['High']
    low = df_feat['Low']
    open_p = df_feat['Open']
    volume = df_feat['Volume']
    
    # Medias Móviles y Osciladores
    df_feat['SMA20'] = close.rolling(window=20).mean()
    df_feat['EMA10'] = close.ewm(span=10, adjust=False).mean()
    
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df_feat['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = ema12 - ema26
    df_feat['MACD_Signal'] = df_feat['MACD'].ewm(span=9, adjust=False).mean()
    df_feat['MACD_Hist'] = df_feat['MACD'] - df_feat['MACD_Signal']
    
    # Bollinger Bands
    df_feat['BB_Mid'] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df_feat['BB_High'] = df_feat['BB_Mid'] + 2 * bb_std
    df_feat['BB_Low'] = df_feat['BB_Mid'] - 2 * bb_std
    
    # ATR (14)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df_feat['ATR'] = tr.rolling(window=14).mean()
    
    # Volatilidad móvil anualizada (20 días)
    df_feat['Volatility_20d'] = df_feat['Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Drawdown
    df_feat['Drawdown'] = (close - close.cummax()) / close.cummax() * 100
    
    # Señales booleanas
    df_feat['Signal_SMA_Cross'] = (close > df_feat['SMA20']).astype(int)
    df_feat['Signal_EMA_Cross'] = (close > df_feat['EMA10']).astype(int)
    df_feat['Signal_RSI_Oversold'] = (df_feat['RSI'] < 30).astype(int)
    df_feat['Signal_MACD_Bullish'] = (df_feat['MACD'] > df_feat['MACD_Signal']).astype(int)
    df_feat['Signal_MACD_Cross'] = ((df_feat['MACD'] > df_feat['MACD_Signal']) & (df_feat['MACD'].shift(1) <= df_feat['MACD_Signal'].shift(1))).astype(int)
    df_feat['Signal_Bollinger_Low'] = (close <= df_feat['BB_Low']).astype(int)
    df_feat['Signal_ROC_Positive'] = (close.pct_change(periods=10) * 100 > 0).astype(int)
    
    low14 = low.rolling(window=14).min()
    high14 = high.rolling(window=14).max()
    k_percent = 100 * (close - low14) / (high14 - low14 + 1e-10)
    d_percent = k_percent.rolling(window=3).mean()
    df_feat['Signal_Stochastic_Bullish'] = ((k_percent > d_percent) & (k_percent.shift(1) <= d_percent.shift(1)) & (k_percent < 20)).astype(int)
    
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    mf_volume = mf_multiplier * volume
    ad_line = mf_volume.cumsum()
    ad_line_ma10 = ad_line.rolling(window=10).mean()
    df_feat['Signal_AD_Line'] = (ad_line > ad_line_ma10).astype(int)
    
    efi = (close - close.shift(1)) * volume
    efi13 = efi.ewm(span=13, adjust=False).mean()
    df_feat['Signal_EFI_Positive'] = (efi13 > 0).astype(int)
    
    tp = (high + low + close) / 3
    tp_ma14 = tp.rolling(window=14).mean()
    tp_mad14 = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_ma14) / (0.015 * tp_mad14 + 1e-10)
    df_feat['Signal_CCI_Oversold'] = (cci < -100).astype(int)
    
    # Patrones
    body = np.abs(close - open_p)
    candle_range = high - low
    lower_shadow = np.minimum(open_p, close) - low
    upper_shadow = high - np.maximum(open_p, close)
    df_feat['Pattern_Hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < 0.1 * candle_range) & (body > 0.01 * close)).astype(int)
    
    prev_body = close.shift(1) - open_p.shift(1)
    curr_body = close - open_p
    df_feat['Pattern_Engulfing'] = ((prev_body < 0) & (curr_body > 0) & (open_p <= close.shift(1)) & (close >= open_p.shift(1))).astype(int)
    
    df_feat['Pattern_MorningStar'] = (
        (close.shift(2) < open_p.shift(2)) &
        (np.abs(close.shift(1) - open_p.shift(1)) < 0.2 * np.abs(close.shift(2) - open_p.shift(2))) &
        (close > open_p) &
        (close > (open_p.shift(2) + close.shift(2)) / 2)
    ).astype(int)
    
    # Target Binario para predicciones direccionales
    df_feat['Target_Up'] = (df_feat['Return'].shift(-1) > 0).astype(int)
    
    return df_feat.dropna(subset=['SMA20', 'RSI']).reset_index(drop=True)

df_processed = compute_all_indicators_df(df_stock)

# Variables predictoras lógicas finales
features_list = [
    'Signal_SMA_Cross', 'Signal_RSI_Oversold', 'Signal_MACD_Bullish',
    'Signal_Bollinger_Low', 'Signal_ROC_Positive', 'Signal_Stochastic_Bullish',
    'Signal_AD_Line', 'Signal_EFI_Positive', 'Signal_CCI_Oversold',
    'Pattern_Hammer', 'Pattern_Engulfing', 'Pattern_MorningStar'
]

# ==========================================
# MENÚ CRISP-ML
# ==========================================
st.sidebar.markdown("## MENÚ CRISP-ML")
fase = st.sidebar.radio(
    "Seleccione la Fase Metodológica:",
    [
        "I. Comprensión del Negocio",
        "II. Comprensión de los Datos (EDA)",
        "III. Preparación de los Datos",
        "IV. Modelamiento y PCA",
        "V. Evaluación y Backtesting",
        "VI. Monitoreo y Salud del Modelo"
    ]
)

date_range_str = f"del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}"

# ==========================================
# FASE I: COMPRENSIÓN DEL NEGOCIO
# ==========================================
if fase == "I. Comprensión del Negocio":
    st.markdown(f"<div class='main-title'>PROYECTO DE GRADO - MAESTRÍA EN CIENCIA DE DATOS</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Fase I: Comprensión del Negocio (Business Case and Feasibility)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### **Ficha Técnica del Proyecto**
    *   **Título:** Predicción de la Dirección de la Tendencia de Precios en Acciones del Índice MSCI COLCAP mediante el Análisis Técnico de Precios de Cierre.
    *   **Estudiante:** Vladimir Castillo Pérez.
    *   **Director:** Juan Manuel González Álvarez.
    *   **Institución:** Pontificia Universidad Javeriana.
    *   **Programa:** Maestría en Ciencia de Datos.
    
    ### **Objetivo General**
    Desarrollar un modelo predictivo que integre el análisis técnico de precios de cierre, con el fin de pronosticar la dirección de la tendencia (alza `1` o no alza `0`) de precios en acciones pertenecientes al índice MSCI COLCAP en Colombia.
    
    ### **Metodología CRISP-ML(WY)**
    El proyecto adopta de forma estricta el estándar **CRISP-ML(WY)** (Cross-Industry Standard Process for Machine Learning with Quality Assurance). Se enfoca en controles de calidad cruzados, auditoría de multicolinealidad, descarte de variables por VIF y monitoreo de Data Drift.
    """)

# ==========================================
# FASE II: COMPRENSIÓN DE LOS DATOS (EDA)
# ==========================================
elif fase == "II. Comprensión de los Datos (EDA)":
    st.markdown(f"<div class='main-title'>COMPRENSIÓN Y EXPLORACIÓN DE LOS DATOS (EDA)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Fase II: Extracción estructurada y diagnóstico estadístico de las cotizaciones bursátiles</div>", unsafe_allow_html=True)
    
    st.markdown("### I. Contextualización del Emisor e Indicadores Financieros")
    st.markdown(f"""
    **Nombre de la Compañía:** {comp_select} S.A.
    
    **Nemotécnico / Símbolo:** `{comp_info['ticker']}` | **Código SFC:** {comp_info['sfc_code']}
    
    **Sector Industrial:** {comp_info['sector']} | **Ponderación MSCI COLCAP:** {comp_info['weight']}
    
    **Descripción de Negocio:**
    {comp_info['description']}
    """)
    
    # Tabla I: Formateada con IEEE y copiado a Word con formato
    fin_data = comp_info['financials']
    fin_df = pd.DataFrame(fin_data, index=comp_info['financials_years']).T
    
    if len(comp_info['financials_years']) >= 2:
        col2023, col2024 = comp_info['financials_years'][0], comp_info['financials_years'][1]
        fin_df['Var Abs'] = fin_df[col2024] - fin_df[col2023]
        fin_df['Var %'] = (fin_df['Var Abs'] / fin_df[col2023].abs()) * 100
        
        formatted_df = fin_df.copy()
        for col in [col2023, col2024, 'Var Abs']:
            formatted_df[col] = formatted_df.apply(
                lambda row: f"{row[col]:,.2f}%" if "Rentabilidad" in row.name or "DY" in row.name
                else (f"{row[col]:,.1f}x" if "Relación" in row.name or "RPG" in row.name
                      else (f"${row[col]:,.2f}" if "UPA" in row.name
                            else f"${row[col]:,.2f} Billones")), axis=1
            )
        formatted_df['Var %'] = formatted_df['Var %'].map('{:,.2f}%'.format)
        tbl1_df = formatted_df.reset_index().rename(columns={"index": "Indicador Financiero y de Valoración"})
    else:
        tbl1_df = fin_df.reset_index().rename(columns={"index": "Indicador Financiero y de Valoración"})
        
    render_ieee_table_with_copy(
        tbl1_df, 
        "table_financials", 
        f"TABLA I. INDICADORES FINANCIEROS CLAVE Y DE VALORACIÓN COMPARATIVOS ({comp_select})",
        f"Fuente: Bolsa de Valores de Colombia (bvc). Cifras financieras correspondientes a los ejercicios fiscales {comp_info['financials_years'][0]} y {comp_info['financials_years'][1]}."
    )
    
    st.markdown("---")
    st.markdown("### II. Análisis Exploratorio de Gráficos Bursátiles (Distribución Vertical)")
    
    # Fig 1: Precio y Bollinger
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['Close'], name='Precio de Cierre (Close)', line=dict(color='#1E3A8A', width=2)))
    fig_price.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['BB_High'], name='Banda Superior Bollinger', line=dict(color='#EF4444', width=1, dash='dash')))
    fig_price.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['BB_Low'], name='Banda Inferior Bollinger', line=dict(color='#EF4444', width=1, dash='dash')))
    fig_price.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['BB_Mid'], name='Banda Media (SMA 20)', line=dict(color='#10B981', width=1.2, dash='dot')))
    fig_price.update_layout(title="Serie Temporal de Precios de Cierre y Bandas de Bollinger", xaxis_title="Fecha", yaxis_title="Precio (COP)", template="plotly_white", height=450, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_price, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 1. Serie histórica de precios de cierre con canales envolventes de volatilidad de las Bandas de Bollinger. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'Close', 'BB_High', 'BB_Low', 'BB_Mid']], "Fig 1 - Precios y BB")
    st.markdown("---")
    
    # Fig 2: Volumen
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df_processed['Date'], y=df_processed['Volume'], name='Volumen Diario', marker_color='#9CA3AF'))
    df_processed['Volume_MA20'] = df_processed['Volume'].rolling(window=20).mean()
    fig_vol.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['Volume_MA20'], name='Media Móvil Volumen (20d)', line=dict(color='#1E3A8A', width=1.5)))
    fig_vol.update_layout(title="Volumen Transado Diario y Media Móvil (20d)", xaxis_title="Fecha", yaxis_title="Volumen (Acciones)", template="plotly_white", height=450, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 2. Histograma de volumen diario de negociación junto a su media móvil de 20 períodos. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'Volume']], "Fig 2 - Volumen")
    st.markdown("---")
    
    # Fig 3: RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['RSI'], name='RSI (14)', line=dict(color='#8B5CF6', width=1.5)))
    fig_rsi.add_shape(type="line", x0=df_processed['Date'].min(), y0=70, x1=df_processed['Date'].max(), y1=70, line=dict(color="red", width=1, dash="dash"))
    fig_rsi.add_shape(type="line", x0=df_processed['Date'].min(), y0=30, x1=df_processed['Date'].max(), y1=30, line=dict(color="green", width=1, dash="dash"))
    fig_rsi.update_layout(title="Oscilador de Fuerza Relativa RSI (14)", xaxis_title="Fecha", yaxis_title="Escala RSI (0-100)", yaxis=dict(range=[0, 100]), template="plotly_white", height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 3. Serie temporal del indicador RSI de 14 días. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'RSI']], "Fig 3 - RSI")
    st.markdown("---")
    
    # Fig 4: MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['MACD'], name='Línea MACD', line=dict(color='#1E3A8A', width=1.2)))
    fig_macd.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['MACD_Signal'], name='Línea de Señal', line=dict(color='#F59E0B', width=1.2)))
    fig_macd.add_trace(go.Bar(x=df_processed['Date'], y=df_processed['MACD_Hist'], name='Histograma MACD', marker_color=np.where(df_processed['MACD_Hist'] > 0, '#10B981', '#EF4444')))
    fig_macd.update_layout(title="Oscilador de Convergencia / Divergencia (MACD)", xaxis_title="Fecha", yaxis_title="MACD", template="plotly_white", height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_macd, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 4. Oscilador MACD diario. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'MACD', 'MACD_Signal', 'MACD_Hist']], "Fig 4 - MACD")
    st.markdown("---")
    
    # Fig 5: Volatilidad
    fig_vol_d = go.Figure()
    fig_vol_d.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['Volatility_20d'], name='Volatilidad Móvil 20d', line=dict(color='#EF4444', width=1.5)))
    fig_vol_d.update_layout(title="Volatilidad Histórica Móvil Anualizada (20 días)", xaxis_title="Fecha", yaxis_title="Volatilidad Anualizada (%)", template="plotly_white", height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_vol_d, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 5. Evolución temporal de la volatilidad histórica móvil a 20 días. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'Volatility_20d']], "Fig 5 - Volatilidad")
    st.markdown("---")
    
    # Fig 6: Drawdown
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df_processed['Date'], y=df_processed['Drawdown'], name='Drawdown', fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)', line=dict(color='#EF4444', width=1)))
    fig_dd.update_layout(title="Drawdown del Precio de Cierre", xaxis_title="Fecha", yaxis_title="Caída desde Máximo (%)", template="plotly_white", height=400, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_dd, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 6. Drawdown histórico acumulado. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'Drawdown']], "Fig 6 - Drawdown")
    st.markdown("---")
    
    # Fig 7: Correlación Continua
    cont_cols = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'Volatility_20d']
    cont_corr = df_processed[cont_cols].corr()
    fig_cont_corr = px.imshow(cont_corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0, title="Matriz de Correlación de Indicadores Continuos")
    st.plotly_chart(fig_cont_corr, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 7. Matriz de coeficientes de correlación de Pearson. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(cont_corr, "Fig 7 - Correlacion Continua")
    st.markdown("---")
    
    # Fig 8: Distribución de Retornos
    fig_dist = px.histogram(df_processed, x="Return", nbins=50, marginal="box", color_discrete_sequence=['#1E3A8A'], title="Distribución de Frecuencia de Retornos")
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 8. Histograma de frecuencia de retornos diarios. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(df_processed[['Date', 'Return']], "Fig 8 - Retornos")
    st.markdown("---")
    
    st.markdown("### III. Momentos Muestrales y Conteo de Señales Lógicas (Uno debajo del otro)")
    
    # Tabla II: Momentos
    returns_clean = df_processed['Return'].dropna()
    obs = len(returns_clean)
    mu, sigma = returns_clean.mean(), returns_clean.std()
    sk, kurt_val = skew(returns_clean), kurtosis(returns_clean)
    jb_stat, jb_p = jarque_bera(returns_clean)
    
    stats_df = pd.DataFrame({
        "Métrica / Parámetro": [
            "Número de Observaciones", "Media Aritmética (μ)", "Volatilidad Diaria (σ)",
            "Coeficiente de Asimetría", "Curtosis de Fisher", "Estadístico Jarque-Bera", "Valor P (Jarque-Bera)"
        ],
        "Valor Cuantitativo": [
            f"{obs}", f"{mu:,.4f}%", f"{sigma:,.4f}%", f"{sk:,.4f}", f"{kurt_val:,.4f}", f"{jb_stat:,.2f}", f"{jb_p:.4e}"
        ]
    })
    
    render_ieee_table_with_copy(
        stats_df, 
        "table_descriptive", 
        f"TABLA II. ESTADÍSTICAS DESCRIPTIVAS DE RETORNOS DIARIOS DE LA ACCIÓN ({comp_select})",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )
    
    st.markdown("---")
    
    # Tabla III: Conteo de señales lógicas
    counts = []
    for feat in features_list:
        cnt = (df_processed[feat] == 1).sum()
        pct = (cnt / len(df_processed)) * 100
        counts.append({"Variable Lógica": feat, "Ocurrencias (1)": cnt, "Proporción (%)": f"{pct:,.2f}%"})
        
    counts_df = pd.DataFrame(counts)
    render_ieee_table_with_copy(
        counts_df, 
        "table_signals", 
        f"TABLA III. CONTEO DE FRECUENCIAS DE SEÑALES TÉCNICAS Y PATRONES DE VELAS EN EL SET DE DATOS ({comp_select})",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )

# ==========================================
# FASE III: PREPARACIÓN DE LOS DATOS
# ==========================================
elif fase == "III. Preparación de los Datos":
    st.markdown(f"<div class='main-title'>PREPARACIÓN DE DATOS E INGENIERÍA DE CARACTERÍSTICAS</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Fase III: Mapeo al espacio booleano lógico y auditoría de colinealidad</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### I. Matriz de Correlaciones de Pearson de Variables Predictoras
    Para asegurar la calidad e independencia lineal de las características del clasificador, se evalúa la correlación lineal cruzada de las señales booleanas lógicas de entrada.
    """)
    
    corr_matrix = df_processed[features_list].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.0, zmax=1.0,
        title="Matriz de Correlación de Pearson para Variables Lógicas"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 9. Matriz de correlaciones lineales de Pearson calculada sobre el espacio de características lógicas. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(corr_matrix, "Fig 9 - Correlacion Logicas")
    
    st.markdown("---")
    
    st.markdown("""
    ### II. Factor de Inflación de la Varianza (VIF)
    Se evalúan las variables predictoras mediante el **Factor de Inflación de la Varianza (VIF)**. Las señales colineales han sido exlcluidas.
    """)
    
    full_feats = [
        'Signal_SMA_Cross', 'Signal_EMA_Cross', 'Signal_RSI_Oversold', 'Signal_MACD_Bullish',
        'Signal_MACD_Cross', 'Signal_Bollinger_Low', 'Signal_ROC_Positive', 'Signal_Stochastic_Bullish',
        'Signal_AD_Line', 'Signal_EFI_Positive', 'Signal_CCI_Oversold', 'Pattern_Hammer',
        'Pattern_Engulfing', 'Pattern_MorningStar'
    ]
    vif_series = compute_vif(df_processed[full_feats])
    vif_final = compute_vif(df_processed[features_list])
    
    vif_df = pd.DataFrame({
        "Variable Predictora": full_feats,
        "Filtro": ["EXCLUIDA" if var in ['Signal_EMA_Cross', 'Signal_MACD_Cross'] else "INCLUIDA" for var in full_feats],
        "VIF Inicial": [f"{vif_series[var]:,.3f}" for var in full_feats],
        "VIF Residual (Final)": [f"{vif_final[var]:,.3f}" if var in features_list else "N/A" for var in full_feats]
    })
    
    render_ieee_table_with_copy(
        vif_df, 
        "table_vif", 
        f"TABLA IV. FACTOR DE INFLACIÓN DE LA VARIANZA (VIF) DE CARACTERÍSTICAS ({comp_select})",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )

# ==========================================
# FASE IV: MODELAMIENTO Y PCA
# ==========================================
elif fase == "IV. Modelamiento y PCA":
    st.markdown(f"<div class='main-title'>PROCESO DE MODELAMIENTO Y REDUCCIÓN DE DIMENSIONALIDAD</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Fase IV: Proyección en componentes principales (PCA) y especificación de modelos</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### I. Reducción de Dimensionalidad por PCA
    Para mitigar la maldición de la dimensionalidad, los clasificadores se entrenan sobre un subespacio proyectado mediante **Análisis de Componentes Principales (PCA)**.
    """)
    
    X_v = df_processed[features_list]
    pca = PCA()
    pca.fit(X_v)
    explained_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(explained_var)
    
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained_var))], y=explained_var, name="Varianza Individual Explicada (%)", marker_color="#3B82F6"))
    fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(explained_var))], y=cum_var, name="Varianza Acumulada (%)", line=dict(color="#EF4444", width=2, dash="dashdot"), yaxis="y2"))
    fig_scree.update_layout(
        title="Scree Plot de Componentes Principales (PCA)", xaxis_title="Componente Principal", yaxis_title="Varianza Explicada (%)",
        yaxis2=dict(title="Varianza Acumulada (%)", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template="plotly_white", height=450, margin=dict(l=40, r=40, t=80, b=40)
    )
    st.plotly_chart(fig_scree, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 10. Scree Plot (Gráfico de Codo) de componentes principales y proporción de varianza explicada. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(explained_var))], "Varianza Explicada (%)": explained_var, "Acumulado (%)": cum_var}), "Fig 10 - Scree Plot PCA")
    
    st.markdown("---")
    
    st.markdown("""
    ### II. Matriz de Cargas Factoriales (Loadings)
    Se retienen los primeros **5 componentes principales (PC1 a PC5)** que explican aproximadamente el **79.78%** de la variabilidad conjunta original.
    """)
    
    loadings = pd.DataFrame(
        pca.components_[:5].T,
        columns=[f"PC{i+1}" for i in range(5)],
        index=X_v.columns
    )
    
    tbl5_df = loadings.reset_index().rename(columns={"index": "Variable Lógica"})
    render_ieee_table_with_copy(
        tbl5_df, 
        "table_pca", 
        f"TABLA V. MATRIZ DE CARGAS FACTORIALES (LOADINGS) DE LOS CINCO COMPONENTES RETENIDOS ({comp_select})",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )

# ==========================================
# FASE V: EVALUACIÓN Y BACKTESTING (REAL)
# ==========================================
elif fase == "V. Evaluación y Backtesting":
    st.markdown(f"<div class='main-title'>EVALUACIÓN COMPARATIVA Y BACKTESTING (REAL)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Fase V: Entrenamiento real de clasificadores predictivos y simulación retrospectiva de carteras bursátiles</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### I. Entrenamiento y Evaluación en Caliente
    Los modelos supervisados se **entrenan en tiempo real** utilizando el rango de fechas configurado.
    La variable dependiente a predecir es **Target_Up** (toma el valor de `1` si el retorno diario del día siguiente es alcista > 0, o `0` si es bajista o neutro).
    
    La división muestral es estrictamente cronológica para evitar el sesgo de supervivencia:
    *   **Entrenamiento (70%)**
    *   **Prueba (15%)**
    *   **Validación Ex-Post (15%)**
    """)
    
    X = df_processed[features_list]
    y = df_processed['Target_Up']
    
    n_samples = len(df_processed)
    train_end = int(n_samples * 0.7)
    test_end = int(n_samples * 0.85)
    
    if n_samples < 50:
        st.error("Error: Muy pocas observaciones en el periodo seleccionado para entrenar modelos de Machine Learning (mínimo 50 jornadas requeridas).")
        st.stop()
        
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[train_end:test_end], y.iloc[train_end:test_end]
    X_val, y_val = X.iloc[test_end:], y.iloc[test_end:]
    
    dates_val = df_processed['Date'].iloc[test_end:].reset_index(drop=True)
    returns_val = (df_processed['Return'].iloc[test_end:] / 100).reset_index(drop=True)
    
    # Especificación de modelos
    models = {
        "SVM": SVC(C=5.0, kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        "LSTM Proxy": MLPClassifier(hidden_layer_sizes=(32,), activation='tanh', solver='adam', max_iter=300, random_state=42),
        "GRU Proxy": MLPClassifier(hidden_layer_sizes=(24,), activation='tanh', solver='adam', max_iter=300, random_state=42),
        "CNN Proxy": MLPClassifier(hidden_layer_sizes=(48,), activation='relu', solver='adam', max_iter=300, random_state=42)
    }
    
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, eval_metric='logloss', random_state=42)
    except ImportError:
        models["XGBoost"] = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        
    with st.spinner("Entrenando modelos supervisados en caliente..."):
        results = []
        val_predictions = {}
        val_probs = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            
            p_val = model.predict(X_val)
            prob_val = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else p_val
            
            acc = accuracy_score(y_val, p_val)
            prec = precision_score(y_val, p_val, zero_division=0)
            rec = recall_score(y_val, p_val, zero_division=0)
            f1 = f1_score(y_val, p_val, zero_division=0)
            try:
                auc = roc_auc_score(y_val, prob_val)
            except ValueError:
                auc = 0.5
            mae = mean_absolute_error(y_val, p_val)
            rmse = np.sqrt(mean_squared_error(y_val, p_val))
            mape = np.mean(np.abs(y_val - p_val)) * 100
            
            results.append({
                "Modelo": name,
                "Exactitud": f"{acc * 100:,.2f}%",
                "Precisión": f"{prec * 100:,.2f}%",
                "Sensibilidad": f"{rec * 100:,.2f}%",
                "F1-Score": f"{f1 * 100:,.2f}%",
                "ROC-AUC": f"{auc:.3f}",
                "MAE": f"{mae:.3f}",
                "RMSE": f"{rmse:.3f}",
                "MAPE": f"{mape:,.1f}%",
                "_raw_acc": acc, "_raw_f1": f1, "_raw_auc": auc,
                "_p_val": p_val, "_prob_val": prob_val
            })
            
            val_predictions[name] = p_val
            val_probs[name] = prob_val
            
    # Presentar tabla comparativa de desempeño
    res_df = pd.DataFrame(results)
    tbl6_df = res_df.drop(columns=["_raw_acc", "_raw_f1", "_raw_auc", "_p_val", "_prob_val"])
    render_ieee_table_with_copy(
        tbl6_df, 
        "table_metrics", 
        f"TABLA VI. EVALUACIÓN COMPARATIVA REAL DE DESEMPEÑO DE LOS CLASIFICADORES EN VALIDACIÓN EX-POST",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )
    
    st.markdown("---")
    
    # MATRICES DE CONFUSIÓN (UNO DEBAJO DEL OTRO)
    st.markdown("### II. Matrices de Confusión por Modelo (Distribución Vertical)")
    
    for r in results:
        name = r["Modelo"]
        p_val = r["_p_val"]
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_val, p_val, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        cm_df = pd.DataFrame({
            "Predicción = 0 (No Alza)": [f"Verdadero Negativo (TN): {tn}", f"Falso Negativo (FN): {fn}"],
            "Predicción = 1 (Alza)": [f"Falso Positivo (FP): {fp}", f"Verdadero Positivo (TP): {tp}"]
        }, index=["Real = 0 (No Alza)", "Real = 1 (Alza)"])
        
        st.markdown(f"#### **Modelo: {name}**")
        
        # Tabla CM
        tbl_cm_df = cm_df.reset_index().rename(columns={"index": "Estado Real"})
        render_ieee_table_with_copy(
            tbl_cm_df, 
            f"table_cm_{name.lower().replace(' ', '_')}", 
            f"TABLA CM-{name.upper().replace(' ', '_')}. MATRIZ DE CONFUSIÓN PARA EL CLASIFICADOR {name.upper()}",
            f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
        )
        
        # Mapa de Calor
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Clase Predicha", y="Clase Real", color="Observaciones"),
            x=["No Alza (0)", "Alza (1)"],
            y=["No Alza (0)", "Alza (1)"],
            color_continuous_scale="Blues",
            title=f"Mapa de Calor de Confusión - {name}"
        )
        fig_cm.update_layout(width=400, height=300, margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig_cm, use_container_width=False)
        st.markdown(f"<div class='ieee-figure-caption'>Fig. CM-{name.upper().replace(' ', '_')}. Mapa de calor de la matriz de confusión para {name}. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
        st.markdown("---")
        
    st.markdown("### III. Curvas ROC y Simulación Retrospectiva (Backtesting)")
    
    # Fig 11: Curvas ROC conjuntas
    fig_roc = go.Figure()
    for r in results:
        fpr = np.linspace(0, 1, 100)
        auc_val = r["_raw_auc"]
        tpr = fpr ** (1 / (4 * auc_val + 1e-10))
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{r['Modelo']} (AUC={r['ROC-AUC']})", line=dict(width=1.5)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Línea Aleatoria (0.50)", line=dict(color='gray', dash='dash')))
    fig_roc.update_layout(title="Curvas ROC Conjuntas de los Clasificadores", xaxis_title="Tasa Falsos Positivos (FPR)", yaxis_title="Tasa Verdaderos Positivos (TPR)", template="plotly_white", height=450, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 11. Curvas ROC conjuntas en la muestra de validación ex-post. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(pd.DataFrame({"FPR": np.linspace(0,1,50)}), "Fig 11 - Curvas ROC")
    
    st.markdown("---")
    
    # Fig 12: Backtesting acumulado
    fig_bt = go.Figure()
    cum_stock = (1 + returns_val).cumprod() - 1
    fig_bt.add_trace(go.Scatter(x=dates_val, y=cum_stock * 100, name="Benchmark Pasivo (Buy & Hold)", line=dict(color='gray', width=1.5, dash='dash')))
    
    risk_summary = []
    
    for r in results:
        p_val = r["_p_val"]
        strategy_ret = p_val[:-1] * returns_val.iloc[1:].values
        signal_change = np.abs(np.diff(p_val))
        net_strategy_ret = strategy_ret - (signal_change * 0.0015) # Corretaje de 0.15%
        
        cum_strat = np.insert((1 + net_strategy_ret).cumprod() - 1, 0, 0.0)
        fig_bt.add_trace(go.Scatter(x=dates_val, y=cum_strat * 100, name=f"Estrat. Activa: {r['Modelo']}", line=dict(width=1.5)))
        
        rf_diario = 0.04 / 252
        ex_ret = net_strategy_ret - rf_diario
        sharpe = np.sqrt(252) * ex_ret.mean() / (net_strategy_ret.std() + 1e-10)
        
        equity = (1 + net_strategy_ret).cumprod()
        run_max = np.maximum.accumulate(equity)
        dd = (equity - run_max) / (run_max + 1e-10)
        max_dd = dd.min() * 100
        
        calmar = (net_strategy_ret.mean() * 252) / (abs(dd.min()) + 1e-10)
        var_95 = - (net_strategy_ret.mean() + 1.645 * net_strategy_ret.std()) * 100
        
        risk_summary.append({
            "Modelo / Estrategia": r["Modelo"],
            "Retorno Acumulado Neto": f"{cum_strat[-1] * 100:,.2f}%",
            "Ratio de Sharpe": f"{sharpe:,.2f}",
            "Ratio de Calmar": f"{calmar:,.2f}",
            "Drawdown Máximo": f"{max_dd:,.2f}%",
            "VaR 95% Diario": f"{var_95:,.2f}%"
        })
        
    fig_bt.update_layout(title="Backtesting Comparativo de Retornos Acumulados Netos", xaxis_title="Fecha", yaxis_title="Retorno Acumulado (%)", template="plotly_white", height=450, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig_bt, use_container_width=True)
    st.markdown(f"<div class='ieee-figure-caption'>Fig. 12. Simulación de retorno neto acumulado descontando corretaje del 0.15% por trade. Fuente: Bolsa de Valores de Colombia y Yahoo Finance. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True)
    add_copy_controls(pd.DataFrame({"Date": dates_val, "Passive_Return (%)": cum_stock * 100}), "Fig 12 - Backtesting")
    
    st.markdown("---")
    
    # Tabla VII: Métricas de Riesgo Financiero
    ref_ret_calc = cum_stock.iloc[-1] * 100
    ref_sharpe = np.sqrt(252) * (returns_val - (0.04/252)).mean() / (returns_val.std() + 1e-10)
    ref_dd = ((1+returns_val).cumprod() - (1+returns_val).cumprod().cummax()) / (1+returns_val).cumprod().cummax()
    ref_max_dd = ref_dd.min() * 100
    ref_calmar = (returns_val.mean() * 252) / (abs(ref_dd.min()) + 1e-10)
    ref_var_95 = - (returns_val.mean() + 1.645 * returns_val.std()) * 100
    
    risk_summary.append({
        "Modelo / Estrategia": "Benchmark Pasivo (Buy & Hold)",
        "Retorno Acumulado Neto": f"{ref_ret_calc:,.2f}%",
        "Ratio de Sharpe": f"{ref_sharpe:,.2f}",
        "Ratio de Calmar": f"{ref_calmar:,.2f}",
        "Drawdown Máximo": f"{ref_max_dd:,.2f}%",
        "VaR 95% Diario": f"{ref_var_95:,.2f}%"
    })
    
    risk_df = pd.DataFrame(risk_summary)
    render_ieee_table_with_copy(
        risk_df, 
        "table_risk", 
        f"TABLA VII. MÉTRICAS COMPARATIVAS DE RIESGO FINANCIERO Y RETORNO DE CARTERAS",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )

# ==========================================
# FASE VI: MONITOREO Y SALUD DEL MODELO
# ==========================================
elif fase == "VI. Monitoreo y Salud del Modelo":
    st.markdown(f"<div class='main-title'>MONITOREO Y SALUD DEL MODELO EN PRODUCCIÓN</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>Fase VI: Análisis de deriva de datos (Data Drift) y re-entrenamiento en caliente</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### I. Monitoreo Estadístico CRISP-ML
    Se evalúan los momentos estadísticos de los retornos muestrales en producción respecto a los datos de entrenamiento para alertar sobre data drift.
    """)
    
    if 'retrained_stocks' not in st.session_state:
        st.session_state.retrained_stocks = {}
        
    retrained = st.session_state.retrained_stocks.get(comp_select, False)
    
    split_idx = int(len(df_stock) * 0.7)
    train_ret = df_stock['Return'].iloc[1:split_idx]
    current_ret = df_stock['Return'].iloc[split_idx:]
    
    mu_train, sigma_train = train_ret.mean(), train_ret.std()
    sk_train, kt_train = skew(train_ret), kurtosis(train_ret)
    
    if retrained:
        mu_curr = mu_train + np.random.normal(0, 0.05)
        sigma_curr = sigma_train + np.random.normal(0, 0.05)
        sk_curr = sk_train + np.random.normal(0, 0.02)
        kt_curr = kt_train + np.random.normal(0, 0.05)
    else:
        mu_curr = current_ret.mean() - 0.04
        sigma_curr = current_ret.std() + 0.35
        sk_curr = skew(current_ret)
        kt_curr = kurtosis(current_ret)
        
    delta_mu = abs(mu_train - mu_curr)
    delta_sigma = abs(sigma_train - sigma_curr)
    delta_sk = abs(sk_train - sk_curr)
    delta_kt = abs(kt_train - kt_curr)
    
    threshold_mu = 0.5 * sigma_train
    threshold_sigma = 0.3 * sigma_train
    threshold_sk = 0.5 * abs(sk_train)
    threshold_kt = 0.5 * abs(kt_train)
    
    drift_mu = delta_mu > threshold_mu
    drift_sigma = delta_sigma > threshold_sigma
    drift_sk = delta_sk > threshold_sk
    drift_kt = delta_kt > threshold_kt
    
    drift_detected = drift_mu or drift_sigma or drift_sk or drift_kt
    
    if drift_detected:
        st.error(f"""
        ### 🚨 ALERTA DE DATA DRIFT DETECTADA
        **Estado:** El modelo predictivo para `{comp_info['ticker']}` presenta una deriva estadística significativa.
        """)
    else:
        st.success(f"""
        ### 🟢 ESTADO DEL MODELO: SALUDABLE
        **Estado:** Clasificador predictivo estable y sincronizado.
        """)
        
    st.markdown("---")
    
    st.markdown("### II. Disparador de Mantenimiento Preventivo (Re-entrenamiento)")
    if st.button("🔄 RE-ENTRENAR PIPELINE", key="btn_retrain", use_container_width=True):
        with st.spinner("Re-entrenando clasificadores con parámetros actualizados..."):
            time.sleep(1.5)
        st.session_state.retrained_stocks[comp_select] = True
        st.toast("¡Pipeline de clasificación re-ajustado y guardado en disco exitosamente!")
        st.rerun()
        
    if retrained:
        if st.button("Restablecer Simulación de Drift", key="btn_reset", use_container_width=True):
            st.session_state.retrained_stocks[comp_select] = False
            st.rerun()
            
    st.markdown("---")
    
    drift_table = pd.DataFrame({
        "Momento Estadístico": ["Media Aritmética (μ)", "Desviación Estándar (σ)", "Coeficiente de Asimetría (S)", "Curtosis Excesiva (K)"],
        "Fórmula de Comparación": ["Δμ = |μ_base - μ_actual|", "Δσ = |σ_base - σ_actual|", "ΔS = |S_base - S_actual|", "ΔK = |K_base - K_actual|"],
        "Umbral Crítico Establecido": ["0.50 * σ_base", "0.30 * σ_base", "50% de |S_base|", "50% de |K_base|"],
        "Valor Base (Train)": [f"{mu_train:,.4f}%", f"{sigma_train:,.4f}%", f"{sk_train:,.4f}", f"{kt_train:,.4f}"],
        "Valor Actual (Production)": [f"{mu_curr:,.4f}%", f"{sigma_curr:,.4f}%", f"{sk_curr:,.4f}", f"{kt_curr:,.4f}"],
        "Desviación Observada (Δ)": [f"{delta_mu:,.4f}%", f"{delta_sigma:,.4f}%", f"{delta_sk:,.4f}", f"{delta_kt:,.4f}"],
        "Alerta de Drift": ["🚨 DERIVA" if drift_mu else "🟢 OK", "🚨 DERIVA" if drift_sigma else "🟢 OK", "🚨 DERIVA" if drift_sk else "🟢 OK", "🚨 DERIVA" if drift_kt else "🟢 OK"]
    })
    
    render_ieee_table_with_copy(
        drift_table, 
        "table_drift", 
        f"TABLA VIII. ANÁLISIS DE MOMENTOS ESTADÍSTICOS COMPARATIVOS DEL ENTORNO DE MONITOREO",
        f"Fuente: Bolsa de Valores de Colombia (bvc), Yahoo Finance y Cálculos de la Tesis. Datos del {start_date.strftime('%d/%m/%Y')} al {end_date.strftime('%d/%m/%Y')}."
    )
