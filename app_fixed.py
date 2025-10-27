# app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_utils import ModelFeatureManager, get_all_cultivos_from_db

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción Agrícola - Corregido",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


class AgriculturalAppFixed:
    def __init__(self):
        self.model_manager = ModelFeatureManager()
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_data = None
        self.load_model_and_data()

    def load_model_and_data(self):
        """Carga el modelo y los datos"""
        try:
            # Cargar modelo corregido
            self.model_data = self.model_manager.load_model_with_features(
                'models/agricultural_production_model_fixed.pkl'
            )

            # Cargar datos para visualizaciones
            self.clima_df, self.siembra_df, self.cosecha_df, self.cultivo_df = self.data_loader.load_all_data()
            self.prepare_visualization_data()

            # Obtener lista completa de cultivos
            self.all_cultivos = get_all_cultivos_from_db(self.data_loader)

            st.success("✅ Modelo y datos cargados correctamente")

        except Exception as e:
            st.error(f"❌ Error cargando el modelo o datos: {e}")
            st.info("Por favor, ejecuta primero el script de entrenamiento corregido.")

    def prepare_visualization_data(self):
        """Prepara datos para visualizaciones"""
        if self.cosecha_df is not None and self.siembra_df is not None:
            self.datos_combinados = self.siembra_df.merge(
                self.cosecha_df,
                on=['id_region', 'id_cultivo', 'anio'],
                suffixes=('_siembra', '_cosecha')
            ).merge(
                self.cultivo_df,
                on='id_cultivo'
            )

            # Calcular rendimiento
            self.datos_combinados['rendimiento'] = (
                    self.datos_combinados['toneladas'] / self.datos_combinados['hectareas']
            )

    def render_sidebar(self):
        """Renderiza la barra lateral"""
        st.sidebar.title("🌱 Navegación")
        app_mode = st.sidebar.selectbox(
            "Selecciona una sección:",
            ["🏠 Dashboard", "📊 Análisis de Datos", "🔮 Predicciones", "📈 Tendencias", "ℹ️ Info del Modelo"]
        )

        st.sidebar.markdown("---")

        # Información del modelo cargado
        if self.model_data is not None:
            st.sidebar.info(f"""
            **Modelo Cargado:**
            - ✅ Características: {len(self.model_manager.expected_features)}
            - ✅ Cultivos: {len(self.model_manager.cultivo_columns)}
            - ✅ Regiones: {len(self.model_manager.region_columns)}
            """)

        st.sidebar.markdown("---")
        st.sidebar.info(
            "Sistema de predicción de producción agrícola - Versión Corregida"
        )

        return app_mode

    def render_model_info(self):
        """Muestra información detallada del modelo"""
        st.title("ℹ️ Información del Modelo")

        if self.model_data is None:
            st.warning("No se pudo cargar la información del modelo.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total de Características", len(self.model_manager.expected_features))

        with col2:
            st.metric("Cultivos en el Modelo", len(self.model_manager.cultivo_columns))

        with col3:
            st.metric("Regiones en el Modelo", len(self.model_manager.region_columns))

        st.markdown("---")

        # Mostrar cultivos disponibles en el modelo
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌾 Cultivos en el Modelo")
            cultivos_modelo = [col.replace('cultivo_', '') for col in self.model_manager.cultivo_columns]
            st.write(f"**{len(cultivos_modelo)} cultivos:**")
            st.write(", ".join(sorted(cultivos_modelo)[:20]) + "..." if len(cultivos_modelo) > 20 else "")

        with col2:
            st.subheader("🏞️ Regiones en el Modelo")
            regiones_modelo = [col.replace('region_', '') for col in self.model_manager.region_columns]
            st.write(f"**{len(regiones_modelo)} regiones:**")
            st.write(", ".join(sorted(regiones_modelo)))

        # Características numéricas
        st.subheader("📊 Características Numéricas")
        st.write(f"**{len(self.model_manager.numeric_columns)} características:**")
        st.write(", ".join(self.model_manager.numeric_columns))

    def render_predictions_fixed(self):
        """Sección de predicciones corregida"""
        st.title("🔮 Predicción de Producción Agrícola - Corregido")

        if self.model_data is None:
            st.error("❌ El modelo no está cargado. No se pueden hacer predicciones.")
            return

        st.success("✅ Modelo cargado correctamente. Puedes realizar predicciones.")

        # Mostrar advertencia sobre cultivos no vistos
        if hasattr(self, 'all_cultivos'):
            cultivos_modelo = [col.replace('cultivo_', '') for col in self.model_manager.cultivo_columns]
            cultivos_no_en_modelo = [c for c in self.all_cultivos if c not in cultivos_modelo]

            if cultivos_no_en_modelo:
                st.warning(f"""
                **Nota:** {len(cultivos_no_en_modelo)} cultivos no están en el modelo entrenado.
                Si seleccionas uno de estos, se usará el cultivo más similar disponible.
                """)

        # Formulario de entrada
        with st.form("prediction_form_fixed"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Parámetros de Siembra")

                # Selector de región
                region_options = [int(col.replace('region_', ''))
                                  for col in self.model_manager.region_columns]
                region = st.selectbox(
                    "Región *",
                    options=sorted(region_options),
                    help="Regiones disponibles en el modelo entrenado"
                )

                # Selector de cultivo
                cultivo_options = [col.replace('cultivo_', '')
                                   for col in self.model_manager.cultivo_columns]
                cultivo = st.selectbox(
                    "Cultivo *",
                    options=sorted(cultivo_options),
                    help="Cultivos disponibles en el modelo entrenado"
                )

                anio = st.number_input(
                    "Año de siembra",
                    min_value=2024,
                    max_value=2030,
                    value=2024
                )

                hectareas = st.number_input(
                    "Hectáreas sembradas *",
                    min_value=1.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0
                )

            with col2:
                st.subheader("🌤️ Condiciones Climáticas Esperadas")

                precip_total = st.slider(
                    "Precipitación total anual (mm) *",
                    min_value=0.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0
                )

                temp_max_promedio = st.slider(
                    "Temperatura máxima promedio (°C) *",
                    min_value=10.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0
                )

                temp_min_promedio = st.slider(
                    "Temperatura mínima promedio (°C) *",
                    min_value=0.0,
                    max_value=25.0,
                    value=12.0,
                    step=1.0
                )

                dias_helada = st.slider(
                    "Días con helada esperados *",
                    min_value=0,
                    max_value=100,
                    value=5
                )

                # Características climáticas adicionales (opcionales)
                st.subheader("🌡️ Características Adicionales (Opcionales)")

                precip_promedio = st.number_input(
                    "Precipitación promedio mensual (mm)",
                    min_value=0.0,
                    max_value=500.0,
                    value=66.7,
                    step=5.0
                )

                grados_dia_total = st.number_input(
                    "Grados día de crecimiento total",
                    min_value=0.0,
                    max_value=5000.0,
                    value=2500.0,
                    step=100.0
                )

            st.markdown("**\* Campos requeridos**")

            # Botón de predicción
            submitted = st.form_submit_button("🎯 Predecir Producción", use_container_width=True)

            if submitted:
                self.make_prediction_fixed(
                    region, cultivo, anio, hectareas,
                    precip_total, temp_max_promedio, temp_min_promedio,
                    dias_helada, precip_promedio, grados_dia_total
                )

    def make_prediction_fixed(self, region, cultivo, anio, hectareas,
                              precip_total, temp_max_promedio, temp_min_promedio,
                              dias_helada, precip_promedio, grados_dia_total):
        """Realiza la predicción usando el método corregido"""
        try:
            # Preparar datos de entrada
            input_data = pd.DataFrame({
                'hectareas': [hectareas],
                'precip_total': [precip_total],
                'precip_promedio': [precip_promedio],
                'temp_max_promedio': [temp_max_promedio],
                'temp_min_promedio': [temp_min_promedio],
                'dias_helada': [dias_helada],
                'grados_dia_total': [grados_dia_total]
            })

            # Preparar características para predicción
            prediction_features = self.model_manager.prepare_prediction_features(
                input_data, cultivo, region
            )

            # Realizar predicción
            X_pred_scaled = self.model_data['scaler'].transform(prediction_features)
            prediction = self.model_data['model'].predict(X_pred_scaled)[0]

            # Calcular rendimiento
            rendimiento = prediction / hectareas

            # Mostrar resultados
            self.display_prediction_results(prediction, rendimiento, hectareas, cultivo, region)

        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            st.info("Por favor, verifica que todos los campos requeridos estén completos.")

    def display_prediction_results(self, prediction, rendimiento, hectareas, cultivo, region):
        """Muestra los resultados de la predicción"""
        st.markdown("---")
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

        st.subheader("📊 Resultados de la Predicción")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Producción Predicha",
                f"{prediction:,.2f} ton",
                delta=f"Rendimiento: {rendimiento:.2f} ton/ha"
            )

        with col2:
            st.metric("Área Sembrada", f"{hectareas:,.1f} ha")

        with col3:
            st.metric("Eficiencia", f"{(rendimiento / hectareas * 1000):.1f}%" if hectareas > 0 else "N/A")

        # Información adicional
        st.write(f"**Cultivo:** {cultivo}")
        st.write(f"**Región:** {region}")
        st.write(f"**Rendimiento estimado:** {rendimiento:.2f} toneladas por hectárea")

        st.markdown('</div>', unsafe_allow_html=True)

        # Gráficos de resultados
        self.render_prediction_charts(prediction, rendimiento, hectareas)

    def render_prediction_charts(self, prediction, rendimiento, hectareas):
        """Renderiza gráficos para los resultados de predicción"""
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de gauge para rendimiento
            if hasattr(self, 'datos_combinados') and not self.datos_combinados.empty:
                rendimiento_promedio = self.datos_combinados['rendimiento'].mean()

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=rendimiento,
                    delta={'reference': rendimiento_promedio},
                    gauge={
                        'axis': {'range': [None, max(rendimiento * 1.5, rendimiento_promedio * 1.5)]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, rendimiento_promedio], 'color': "lightgray"},
                            {'range': [rendimiento_promedio, rendimiento_promedio * 1.5], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': rendimiento_promedio
                        }
                    },
                    title={'text': "Rendimiento vs Promedio Histórico"}
                ))

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gráfico de barras simple
            fig = px.bar(
                x=['Producción Predicha'],
                y=[prediction],
                title="Producción Total Estimada",
                labels={'x': '', 'y': 'Toneladas'},
                color_discrete_sequence=['#2E8B57']
            )
            st.plotly_chart(fig, use_container_width=True)

    # Los demás métodos (render_dashboard, render_data_analysis, etc.) se mantienen igual
    # que en la versión anterior...


def main():
    app = AgriculturalAppFixed()
    app_mode = app.render_sidebar()

    if app_mode == "🏠 Dashboard":
        if hasattr(app, 'render_dashboard'):
            app.render_dashboard()
        else:
            st.title("🏠 Dashboard")
            st.info("Esta funcionalidad está en desarrollo.")

    elif app_mode == "📊 Análisis de Datos":
        if hasattr(app, 'render_data_analysis'):
            app.render_data_analysis()
        else:
            st.title("📊 Análisis de Datos")
            st.info("Esta funcionalidad está en desarrollo.")

    elif app_mode == "🔮 Predicciones":
        app.render_predictions_fixed()

    elif app_mode == "📈 Tendencias":
        if hasattr(app, 'render_trends'):
            app.render_trends()
        else:
            st.title("📈 Tendencias")
            st.info("Esta funcionalidad está en desarrollo.")

    elif app_mode == "ℹ️ Info del Modelo":
        app.render_model_info()


if __name__ == "__main__":
    main()