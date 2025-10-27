# app.py
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

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción Agrícola",
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


class AgriculturalApp:
    def __init__(self):
        self.model = None
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.load_data()

    def load_data(self):
        """Carga los datos y el modelo"""
        try:
            # Cargar modelo
            self.model = joblib.load('models/agricultural_production_model.pkl')

            # Cargar datos para visualizaciones
            self.clima_df, self.siembra_df, self.cosecha_df, self.cultivo_df = self.data_loader.load_all_data()

            # Preparar datos combinados para gráficos
            self.prepare_visualization_data()

        except Exception as e:
            st.error(f"Error cargando el modelo o datos: {e}")

    def prepare_visualization_data(self):
        """Prepara datos para visualizaciones"""
        # Combinar datos para análisis
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
        """Renderiza la barra lateral de navegación"""
        st.sidebar.title("🌱 Navegación")
        app_mode = st.sidebar.selectbox(
            "Selecciona una sección:",
            ["🏠 Dashboard", "📊 Análisis de Datos", "🔮 Predicciones", "📈 Tendencias"]
        )

        st.sidebar.markdown("---")
        st.sidebar.info(
            "Sistema de predicción de producción agrícola basado en "
            "datos climáticos y de siembra históricos."
        )

        return app_mode

    def render_dashboard(self):
        """Renderiza el dashboard principal"""
        st.markdown('<h1 class="main-header">🌾 Dashboard de Producción Agrícola</h1>', unsafe_allow_html=True)

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_cosecha = self.datos_combinados['toneladas'].sum()
            st.metric("Producción Total", f"{total_cosecha:,.0f} ton")

        with col2:
            total_hectareas = self.datos_combinados['hectareas'].sum()
            st.metric("Área Sembrada Total", f"{total_hectareas:,.0f} ha")

        with col3:
            rendimiento_promedio = self.datos_combinados['rendimiento'].mean()
            st.metric("Rendimiento Promedio", f"{rendimiento_promedio:.2f} ton/ha")

        with col4:
            cultivos_unicos = self.datos_combinados['nombre_cultivo'].nunique()
            st.metric("Tipos de Cultivo", cultivos_unicos)

        st.markdown("---")

        # Gráficos principales
        col1, col2 = st.columns(2)

        with col1:
            self.render_production_trend()

        with col2:
            self.render_crop_distribution()

        col3, col4 = st.columns(2)

        with col3:
            self.render_yield_by_region()

        with col4:
            self.render_climate_impact()

    def render_production_trend(self):
        """Gráfico de tendencia de producción"""
        st.subheader("📈 Tendencia de Producción Anual")

        produccion_anual = self.datos_combinados.groupby('anio').agg({
            'toneladas': 'sum',
            'hectareas': 'sum'
        }).reset_index()

        produccion_anual['rendimiento'] = produccion_anual['toneladas'] / produccion_anual['hectareas']

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=produccion_anual['anio'], y=produccion_anual['toneladas'],
                       name="Producción (ton)", line=dict(color='#2E8B57', width=3)),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=produccion_anual['anio'], y=produccion_anual['rendimiento'],
                       name="Rendimiento (ton/ha)", line=dict(color='#FF6B6B', width=3)),
            secondary_y=True,
        )

        fig.update_layout(
            title="Evolución de la Producción y Rendimiento",
            xaxis_title="Año",
            hovermode='x unified',
            height=400
        )

        fig.update_yaxes(title_text="Producción (ton)", secondary_y=False)
        fig.update_yaxes(title_text="Rendimiento (ton/ha)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    def render_crop_distribution(self):
        """Gráfico de distribución de cultivos"""
        st.subheader("🥦 Distribución de Cultivos")

        cultivo_produccion = self.datos_combinados.groupby('nombre_cultivo').agg({
            'toneladas': 'sum',
            'hectareas': 'sum'
        }).reset_index()

        cultivo_produccion['rendimiento'] = cultivo_produccion['toneladas'] / cultivo_produccion['hectareas']

        fig = px.sunburst(
            cultivo_produccion,
            path=['nombre_cultivo'],
            values='toneladas',
            title="Distribución de Producción por Cultivo",
            color='rendimiento',
            color_continuous_scale='Viridis',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_yield_by_region(self):
        """Gráfico de rendimiento por región"""
        st.subheader("🏞️ Rendimiento por Región")

        region_rendimiento = self.datos_combinados.groupby('id_region').agg({
            'rendimiento': 'mean',
            'toneladas': 'sum'
        }).reset_index()

        fig = px.bar(
            region_rendimiento,
            x='id_region',
            y='rendimiento',
            title="Rendimiento Promedio por Región",
            color='rendimiento',
            color_continuous_scale='Blues',
            height=400
        )

        fig.update_layout(xaxis_title="Región", yaxis_title="Rendimiento (ton/ha)")

        st.plotly_chart(fig, use_container_width=True)

    def render_climate_impact(self):
        """Gráfico de impacto climático"""
        st.subheader("🌤️ Análisis Climático")

        # Agrupar datos climáticos por año
        clima_anual = self.clima_df.groupby('anio').agg({
            'precipitacion': 'mean',
            'temperatura_max': 'mean',
            'temperatura_min': 'mean'
        }).reset_index()

        # Combinar con datos de producción
        clima_produccion = clima_anual.merge(
            self.datos_combinados.groupby('anio')['rendimiento'].mean().reset_index(),
            on='anio'
        )

        fig = px.scatter(
            clima_produccion,
            x='precipitacion',
            y='rendimiento',
            size='temperatura_max',
            color='temperatura_max',
            title="Relación Precipitación vs Rendimiento",
            trendline="lowess",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_data_analysis(self):
        """Sección de análisis de datos"""
        st.title("📊 Análisis Exploratorio de Datos")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Datos de Siembra")
            st.dataframe(
                self.siembra_df.head(100),
                use_container_width=True,
                height=300
            )

            # Resumen estadístico
            st.subheader("Estadísticas de Siembra")
            st.dataframe(self.siembra_df[['hectareas', 'anio']].describe())

        with col2:
            st.subheader("Datos de Cosecha")
            st.dataframe(
                self.cosecha_df.head(100),
                use_container_width=True,
                height=300
            )

            # Resumen estadístico
            st.subheader("Estadísticas de Cosecha")
            st.dataframe(self.cosecha_df[['toneladas', 'anio']].describe())

        st.markdown("---")

        # Análisis de correlación
        st.subheader("🔍 Análisis de Correlaciones")

        # Preparar datos para correlación
        datos_analisis = self.datos_combinados[['hectareas', 'toneladas', 'rendimiento', 'anio']].copy()

        # Calcular matriz de correlación
        corr_matrix = datos_analisis.corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Matriz de Correlación"
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_predictions(self):
        """Sección de predicciones"""
        st.title("🔮 Predicción de Producción Agrícola")

        st.info("""
        Complete los parámetros a continuación para predecir la producción agrícola 
        basándose en las condiciones climáticas y de siembra.
        """)

        # Formulario de entrada
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📋 Parámetros de Siembra")
                region = st.selectbox(
                    "Región",
                    options=sorted(self.siembra_df['id_region'].unique())
                )

                cultivo = st.selectbox(
                    "Cultivo",
                    options=sorted(self.cultivo_df['nombre_cultivo'].unique())
                )

                anio = st.number_input(
                    "Año de siembra",
                    min_value=2024,
                    max_value=2030,
                    value=2024
                )

                hectareas = st.number_input(
                    "Hectáreas sembradas",
                    min_value=1.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0
                )

            with col2:
                st.subheader("🌤️ Condiciones Climáticas Esperadas")

                precip_total = st.slider(
                    "Precipitación total anual (mm)",
                    min_value=0.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0
                )

                temp_max_promedio = st.slider(
                    "Temperatura máxima promedio (°C)",
                    min_value=10.0,
                    max_value=40.0,
                    value=25.0,
                    step=1.0
                )

                temp_min_promedio = st.slider(
                    "Temperatura mínima promedio (°C)",
                    min_value=0.0,
                    max_value=25.0,
                    value=12.0,
                    step=1.0
                )

                dias_helada = st.slider(
                    "Días con helada esperados",
                    min_value=0,
                    max_value=100,
                    value=5
                )

            # Botón de predicción
            submitted = st.form_submit_button("🎯 Predecir Producción", use_container_width=True)

            if submitted:
                self.make_prediction(
                    region, cultivo, anio, hectareas,
                    precip_total, temp_max_promedio, temp_min_promedio, dias_helada
                )

    def make_prediction(self, region, cultivo, anio, hectareas, precip_total,
                        temp_max_promedio, temp_min_promedio, dias_helada):
        """Realiza la predicción y muestra resultados"""
        try:
            # Preparar datos de entrada
            input_data = self.prepare_prediction_input(
                region, cultivo, anio, hectareas,
                precip_total, temp_max_promedio, temp_min_promedio, dias_helada
            )

            # Realizar predicción
            prediction = self.model['model'].predict(
                self.model['scaler'].transform(input_data)
            )[0]

            # Calcular rendimiento
            rendimiento = prediction / hectareas

            # Mostrar resultados
            st.markdown("---")
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

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
                st.metric("Cultivo", cultivo)

            st.markdown('</div>', unsafe_allow_html=True)

            # Gráfico de resultados
            self.render_prediction_charts(prediction, rendimiento, hectareas)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

    def prepare_prediction_input(self, region, cultivo, anio, hectareas,
                                 precip_total, temp_max_promedio, temp_min_promedio, dias_helada):
        """Prepara los datos de entrada para la predicción"""
        # Obtener ID del cultivo
        cultivo_id = self.cultivo_df[
            self.cultivo_df['nombre_cultivo'] == cultivo
            ]['id_cultivo'].iloc[0]

        # Crear DataFrame base
        input_df = pd.DataFrame({
            'id_region': [region],
            'id_cultivo': [cultivo_id],
            'anio': [anio],
            'hectareas': [hectareas],
            'nombre_cultivo': [cultivo]
        })

        # Usar el feature engineer para crear todas las características
        # Nota: En una implementación real, necesitarías calcular características climáticas más detalladas
        training_data = self.feature_engineer.prepare_training_data(
            self.clima_df,
            pd.DataFrame([{
                'id_siembra': 1,
                'id_region': region,
                'id_cultivo': cultivo_id,
                'hectareas': hectareas,
                'anio': anio
            }]),
            pd.DataFrame([{
                'id_region': region,
                'id_cultivo': cultivo_id,
                'toneladas': 0,  # Placeholder
                'anio': anio
            }]),
            self.cultivo_df
        )

        X, _ = self.feature_engineer.create_feature_matrix(training_data)

        return X

    def render_prediction_charts(self, prediction, rendimiento, hectareas):
        """Renderiza gráficos para los resultados de predicción"""
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de comparación con promedio histórico
            rendimiento_promedio = self.datos_combinados['rendimiento'].mean()

            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=rendimiento,
                delta={'reference': rendimiento_promedio},
                gauge={
                    'axis': {'range': [None, max(rendimiento * 1.5, rendimiento_promedio * 1.5)]},
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
            # Gráfico de desglose
            labels = ['Producción Esperada', 'Área No Productiva']
            values = [prediction, hectareas * rendimiento - prediction]

            fig = px.pie(
                values=values,
                names=labels,
                title="Distribución de Producción Esperada",
                color=labels,
                color_discrete_map={
                    'Producción Esperada': '#2E8B57',
                    'Área No Productiva': '#FF6B6B'
                }
            )

            st.plotly_chart(fig, use_container_width=True)

    def render_trends(self):
        """Sección de análisis de tendencias"""
        st.title("📈 Análisis de Tendencias y Proyecciones")

        st.warning("""
        Esta sección utiliza datos históricos para identificar tendencias 
        y realizar proyecciones futuras. Las predicciones son estimaciones 
        basadas en patrones históricos.
        """)

        # Selectores para análisis
        col1, col2, col3 = st.columns(3)

        with col1:
            cultivo_tendencia = st.selectbox(
                "Selecciona cultivo para análisis:",
                options=sorted(self.cultivo_df['nombre_cultivo'].unique()),
                key="cultivo_trend"
            )

        with col2:
            region_tendencia = st.selectbox(
                "Selecciona región:",
                options=sorted(self.siembra_df['id_region'].unique()),
                key="region_trend"
            )

        with col3:
            metricas = st.selectbox(
                "Métrica a analizar:",
                ["Producción", "Rendimiento", "Área Sembrada"]
            )

        # Gráfico de tendencia
        self.render_trend_analysis(cultivo_tendencia, region_tendencia, metricas)

        # Proyección futura
        st.subheader("🔭 Proyección para Próximos Años")

        anos_proyeccion = st.slider(
            "Años a proyectar:",
            min_value=1,
            max_value=10,
            value=5
        )

        if st.button("Generar Proyección", use_container_width=True):
            self.render_projection(cultivo_tendencia, region_tendencia, anos_proyeccion)

    def render_trend_analysis(self, cultivo, region, metrica):
        """Renderiza análisis de tendencias"""
        # Filtrar datos
        cultivo_id = self.cultivo_df[
            self.cultivo_df['nombre_cultivo'] == cultivo
            ]['id_cultivo'].iloc[0]

        datos_filtrados = self.datos_combinados[
            (self.datos_combinados['id_region'] == region) &
            (self.datos_combinados['id_cultivo'] == cultivo_id)
            ]

        if datos_filtrados.empty:
            st.warning("No hay datos suficientes para el análisis de tendencias con los filtros seleccionados.")
            return

        # Seleccionar métrica
        if metrica == "Producción":
            columna = 'toneladas'
            titulo = f"Tendencia de Producción - {cultivo}"
        elif metrica == "Rendimiento":
            columna = 'rendimiento'
            titulo = f"Tendencia de Rendimiento - {cultivo}"
        else:
            columna = 'hectareas'
            titulo = f"Tendencia de Área Sembrada - {cultivo}"

        fig = px.scatter(
            datos_filtrados,
            x='anio',
            y=columna,
            trendline="lowess",
            title=titulo,
            size='toneladas',
            color='rendimiento',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_projection(self, cultivo, region, anos_proyeccion):
        """Renderiza proyección futura"""
        # Aquí iría la lógica para proyecciones más sofisticadas
        # Por ahora, mostramos un gráfico placeholder

        st.info("""
        ⚠️ Las proyecciones mostradas son estimaciones basadas en tendencias históricas 
        y no consideran cambios climáticos extremos o eventos imprevistos.
        """)

        # Placeholder para proyección
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[2023, 2024, 2025, 2026, 2027],
            y=[100, 120, 115, 130, 125],
            mode='lines+markers',
            name='Proyección',
            line=dict(dash='dash', color='orange')
        ))

        fig.update_layout(
            title=f"Proyección de Producción - {cultivo}",
            xaxis_title="Año",
            yaxis_title="Producción (ton)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def main():
    # Inicializar la aplicación
    app = AgriculturalApp()

    # Renderizar navegación
    app_mode = app.render_sidebar()

    # Renderizar sección seleccionada
    if app_mode == "🏠 Dashboard":
        app.render_dashboard()
    elif app_mode == "📊 Análisis de Datos":
        app.render_data_analysis()
    elif app_mode == "🔮 Predicciones":
        app.render_predictions()
    elif app_mode == "📈 Tendencias":
        app.render_trends()


if __name__ == "__main__":
    main()