# agricultural_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class AgriculturalProductionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False

    def initialize_model(self):
        """Inicializa el modelo Random Forest"""
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X, y, test_size=0.2):
        """Entrena el modelo"""
        print("Iniciando entrenamiento del modelo...")

        if self.model is None:
            self.initialize_model()

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}")
        print(f"Tamaño del conjunto de prueba: {X_test.shape}")

        # Escalar características (opcional para Random Forest, pero útil para comparar)
        self.feature_columns = X.columns.tolist()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entrenar modelo
        print("Entrenando modelo...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluar modelo
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        # Predicciones
        y_pred = self.model.predict(X_test_scaled)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("\n=== RESULTADOS DEL MODELO ===")
        print(f"R² Train: {train_score:.4f}")
        print(f"R² Test: {test_score:.4f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")

        # Validación cruzada
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"Validación Cruzada R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.is_trained = True

        return {
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_scores': cv_scores
        }

    def predict(self, X):
        """Realiza predicciones"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def feature_importance(self, top_n=15):
        """Muestra la importancia de las características"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Graficar
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_imp_df.head(top_n), x='importance', y='feature')
        plt.title(f'Top {top_n} Características Más Importantes')
        plt.tight_layout()
        plt.show()

        return feature_imp_df

    def save_model(self, filepath):
        """Guarda el modelo entrenado"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath):
        """Carga un modelo guardado"""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.scaler = loaded_data['scaler']
        self.feature_columns = loaded_data['feature_columns']
        self.is_trained = True
        print(f"Modelo cargado desde: {filepath}")