# main.py
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from agricultural_model import AgriculturalProductionModel
import pandas as pd
import os


def main():
    print("=== SISTEMA DE PREDICCIÓN DE PRODUCCIÓN AGRÍCOLA ===\n")

    # Paso 1: Cargar datos
    print("Paso 1: Cargando datos desde la base de datos...")
    data_loader = DataLoader()
    clima_df, siembra_df, cosecha_df, cultivo_df = data_loader.load_all_data()

    # Verificar que tenemos datos
    if clima_df is None or siembra_df is None or cosecha_df is None:
        print("Error: No se pudieron cargar todos los datos")
        return

    print(f"Datos climáticos: {clima_df.shape}")
    print(f"Datos de siembra: {siembra_df.shape}")
    print(f"Datos de cosecha: {cosecha_df.shape}")
    print(f"Datos de cultivos: {cultivo_df.shape}")

    # Paso 2: Ingeniería de características
    print("\nPaso 2: Realizando ingeniería de características...")
    feature_engineer = FeatureEngineer()
    training_data = feature_engineer.prepare_training_data(
        clima_df, siembra_df, cosecha_df, cultivo_df
    )

    print(f"Datos de entrenamiento combinados: {training_data.shape}")

    # Crear matriz de características
    X, y = feature_engineer.create_feature_matrix(training_data)
    print(f"Matriz de características: {X.shape}")
    print(f"Variable objetivo: {y.shape}")

    # Paso 3: Entrenar modelo
    print("\nPaso 3: Entrenando modelo...")
    model = AgriculturalProductionModel()
    results = model.train(X, y)

    # Paso 4: Mostrar importancia de características
    print("\nPaso 4: Analizando importancia de características...")
    feature_importance_df = model.feature_importance()
    print("\nTop 10 características más importantes:")
    print(feature_importance_df.head(10))

    # Paso 5: Guardar modelo
    print("\nPaso 5: Guardando modelo...")
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save_model('models/agricultural_production_model.pkl')

    # Paso 6: Ejemplo de predicción
    print("\nPaso 6: Realizando predicción de ejemplo...")
    try:
        # Predecir con los primeros 5 ejemplos del conjunto de prueba
        sample_prediction = model.predict(X.head(5))
        actual_values = y.head(5).values

        comparison_df = pd.DataFrame({
            'Actual': actual_values,
            'Predicho': sample_prediction,
            'Diferencia': abs(actual_values - sample_prediction)
        })

        print("\nEjemplo de predicciones:")
        print(comparison_df)

    except Exception as e:
        print(f"Error en predicción de ejemplo: {e}")

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("El modelo está listo para usar!")


if __name__ == "__main__":
    main()