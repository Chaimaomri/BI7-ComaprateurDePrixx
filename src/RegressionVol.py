import pandas as pd
from google.colab import files
uploaded = files.upload()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Dim_Vols.csv', encoding='ISO-8859-1')

print(df.columns)

# Convertir les colonnes 'DepartHour' et 'ArriveHour' en format datetime
df['DepartHour'] = pd.to_datetime(df['DepartHour'], format='%H:%M')
df['ArriveHour'] = pd.to_datetime(df['ArriveHour'], format='%H:%M')

# Convertir les colonnes 'DepartHour' et 'ArriveHour' en format datetime
df['DepartHour'] = pd.to_datetime(df['DepartHour'], format='%H:%M')
df['ArriveHour'] = pd.to_datetime(df['ArriveHour'], format='%H:%M')

# Si les vols durent moins de 24h, on peut faire :
df['FlightDuration'] = (df['ArriveHour'] - df['DepartHour']).dt.total_seconds() / 3600


# Vérifier les résultats
print(df[['DepartHour', 'ArriveHour', 'FlightDuration']].head())

print(df.columns.tolist())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sélection des variables pertinentes pour la régression
X = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport']]  # Ajoute les caractéristiques souhaitées
y = df['FlightDuration']

# Sélection des variables pertinentes pour la régression
X = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport']]  # Ajoute les caractéristiques souhaitées
y = df['FlightDuration']

# Convertir les variables catégorielles en variables numériques (par exemple, avec des dummies)
X = pd.get_dummies(X, drop_first=True)


# Séparer les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convertir DepartHour et ArriveHour en valeurs numériques (nombre d'heures depuis minuit)
df['DepartHour'] = pd.to_datetime(df['DepartHour']).dt.hour + pd.to_datetime(df['DepartHour']).dt.minute / 60
df['ArriveHour'] = pd.to_datetime(df['ArriveHour']).dt.hour + pd.to_datetime(df['ArriveHour']).dt.minute / 60


# Refaire la séparation des variables X et y
X = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport']]  # Ajoute les caractéristiques souhaitées
y = df['FlightDuration']

# Convertir les variables catégorielles en variables numériques (par exemple, avec des dummies)
X = pd.get_dummies(X, drop_first=True)


# Séparer les données en jeu d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression linéaire
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)


# Évaluer la performance du modèle
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt

# Graphique de la durée des vols
plt.figure(figsize=(10,6))
plt.hist(df['FlightDuration'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution des Durées des Vols')
plt.xlabel('Durée du Vol (en heures)')
plt.ylabel('Nombre de Vols')
plt.show()


# ⚠️ Correction : Ajouter 24h si la durée est négative (vols arrivant le jour suivant)
df.loc[df['FlightDuration'] < 0, 'FlightDuration'] += 24

# Affichage des premières lignes pour vérification
print(df[['DepartHour', 'ArriveHour', 'FlightDuration']].head())

# Histogramme corrigé
plt.figure(figsize=(10, 6))
plt.hist(df['FlightDuration'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution des Durées des Vols (corrigée)')
plt.xlabel('Durée du Vol (en heures)')
plt.ylabel('Nombre de Vols')
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Liste des modèles
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Support Vector Regressor': SVR()
}


# Évaluation des modèles
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

# Affichage des résultats
results_df = pd.DataFrame(results)
print(results_df)

# Supposons que Random Forest donne les meilleurs résultats
best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

# Prédictions sur tout le dataset (ou X_test selon ton besoin)
df['Predicted_Duration'] = best_model.predict(X)

# Exporter les résultats dans un fichier CSV
df_export = df[['DepartHour', 'Airline', 'DepartAirport', 'ArriveAirport', 'FlightDuration', 'Predicted_Duration']]
df_export.to_csv('predictions_duree_vols.csv', index=False)

