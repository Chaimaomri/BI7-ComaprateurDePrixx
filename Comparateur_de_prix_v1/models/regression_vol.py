from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import io

app = Flask(__name__)

@app.route('/regression_vol', methods=['GET', 'POST'])
def regression_vols():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return render_template('regression_vol.html', results=None, predictions=None)

        # Lire le fichier
        df = pd.read_csv(io.StringIO(file.stream.read().decode('ISO-8859-1')))

        # Conversion des heures
        df['DepartHour'] = pd.to_datetime(df['DepartHour'], format='%H:%M')
        df['ArriveHour'] = pd.to_datetime(df['ArriveHour'], format='%H:%M')

        # Calcul de la durée
        df['FlightDuration'] = (df['ArriveHour'] - df['DepartHour']).dt.total_seconds() / 3600
        df.loc[df['FlightDuration'] < 0, 'FlightDuration'] += 24

        # Conversion de l'heure de départ en nombre
        df['DepartHour'] = df['DepartHour'].dt.hour + df['DepartHour'].dt.minute / 60

        # Sélection des variables
        X = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport']]
        y = df['FlightDuration']

        # Encodage des variables catégorielles
        X = pd.get_dummies(X, drop_first=True)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modèles à tester
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Support Vector Regressor': SVR()
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Modèle': name,
                'MAE': round(mae, 3),
                'RMSE': round(rmse, 3),
                'R²': round(r2, 3)
            })

        # Meilleur modèle = Random Forest
        best_model = RandomForestRegressor(random_state=42)
        best_model.fit(X_train, y_train)
        df['Predicted_Duration'] = best_model.predict(X)

        # Colonnes pour affichage
        display_cols = ['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport', 'FlightDuration', 'Predicted_Duration']
        predictions = df[display_cols].to_dict(orient='records')

        return render_template('regression_result.html', results=results, predictions=predictions)

    return render_template('regression_vol.html', results=None, predictions=None)
