import pandas as pd
from google.colab import files
uploaded = files.upload()


df_temps = pd.read_csv('Dim_Temps.csv', encoding='ISO-8859-1')

df_reservations = pd.read_csv('Fait_Reservations.csv', encoding='ISO-8859-1')

# Jointure sur TempsID pour avoir Semaine_Année, Jour_Férié, etc.
df = df_reservations.merge(df_temps, left_on='Temps_FK', right_on='TempsID', how='left')


# 2. Agrégation par semaine
df_weekly = df.groupby('Semaine_Année').agg({
    'Reservation_PK': 'count',
    'Jour_Férié': 'sum',
    'Mois': 'first',
    'Trimestre': 'first'
}).reset_index().rename(columns={'Reservation_PK': 'reservations'})

# Fixer l'année (par exemple 2024)
année = '2024'

# Nettoyage et conversion
df_weekly['Semaine_Année'] = df_weekly['Semaine_Année'].astype(int)

# Construire les dates avec l'année et le numéro de semaine
df_weekly['ds'] = pd.to_datetime(année + df_weekly['Semaine_Année'].astype(str).str.zfill(2) + '-1', format='%G%V-%u')

# Trier et indexer
df_weekly = df_weekly.sort_values('ds')
df_weekly.set_index('ds', inplace=True)


import matplotlib.pyplot as plt

# 4. Affichage de la série temporelle
plt.figure(figsize=(12,5))
plt.plot(df_weekly['reservations'])
plt.title("Nombre de réservations hebdomadaires")
plt.xlabel("Semaine")
plt.ylabel("Réservations")
plt.grid(True)
plt.show()


from statsmodels.tsa.stattools import adfuller
# 5. Vérification de stationnarité
result = adfuller(df_weekly['reservations'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

from statsmodels.tsa.arima.model import ARIMA

# 6. Modélisation ARIMA
model = ARIMA(df_weekly['reservations'], order=(1,1,1))
model_fit = model.fit()


# 7. Prévision sur 12 semaines
forecast = model_fit.forecast(steps=12)

# 8. Affichage de la prévision
plt.figure(figsize=(12,5))
plt.plot(df_weekly['reservations'], label='Historique')
plt.plot(pd.date_range(start=df_weekly.index[-1], periods=13, freq='W')[1:], forecast, label='Prévision', color='orange')
plt.title("Prévision des réservations hebdomadaires (ARIMA)")
plt.xlabel("Semaine")
plt.ylabel("Réservations")
plt.legend()
plt.grid(True)
plt.show()

print(df_reservations.shape[0])


df_weekly.shape[0]


df_weekly['reservations'].describe()


# Créer un DataFrame pour la prévision
forecast_dates = pd.date_range(start=df_weekly.index[-1], periods=13, freq='W')[1:]
df_forecast = pd.DataFrame({
    'ds': forecast_dates,
    'reservations_prevision': forecast
})

# Remettre les réservations historiques dans une colonne, pas en index
df_historique = df_weekly.reset_index()[['ds', 'reservations']]


# Fusionner historique + prévision dans un seul DataFrame
df_final = pd.concat([df_historique, df_forecast], axis=0).sort_values('ds')

# Exporter au format CSV
df_final.to_csv('prevision_reservations.csv', index=False)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p,d,q)(P,D,Q,s)
# Exemple : SARIMA(1,1,1)(1,1,1,52) pour saisonnalité annuelle hebdo
sarima_model = SARIMAX(df_weekly['reservations'],
                       order=(1,1,1),
                       seasonal_order=(1,1,1,52),  # 52 semaines ≈ 1 an
                       enforce_stationarity=False,
                       enforce_invertibility=False)

sarima_fit = sarima_model.fit()
print(sarima_fit.summary())

# Prévision sur 12 semaines
sarima_forecast = sarima_fit.forecast(steps=12)


# Dates de prévision
forecast_dates = pd.date_range(start=df_weekly.index[-1], periods=13, freq='W')[1:]

plt.figure(figsize=(12,5))
plt.plot(df_weekly['reservations'], label='Historique')
plt.plot(forecast_dates, forecast, label='ARIMA(1,1,1)', linestyle='--', color='orange')
plt.plot(forecast_dates, sarima_forecast, label='SARIMA(1,1,1)(1,1,1,52)', linestyle='-.', color='green')
plt.title("Prévision des réservations hebdomadaires (ARIMA vs SARIMA)")
plt.xlabel("Semaine")
plt.ylabel("Réservations")
plt.legend()
plt.grid(True)
plt.show()


# Dates de prévision
forecast_dates = pd.date_range(start=df_weekly.index[-1], periods=13, freq='W')[1:]

# Créer un DataFrame pour la prévision SARIMA
df_sarima = pd.DataFrame({
    'ds': forecast_dates,
    'prevision_sarima': sarima_forecast
})

# Historique
df_historique = df_weekly.reset_index()[['ds', 'reservations']]

# Fusionner historique + prévision
df_export = pd.concat([df_historique, df_sarima], axis=0).sort_values('ds')


df_export.to_csv('prevision_sarima_reservations.csv', index=False)

