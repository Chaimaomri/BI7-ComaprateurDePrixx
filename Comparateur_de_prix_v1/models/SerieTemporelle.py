import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import base64
from io import BytesIO

def run_forecasting(df_reservations, df_temps):
    df = df_reservations.merge(df_temps, left_on='Temps_FK', right_on='TempsID', how='left')
    df_weekly = df.groupby('Semaine_Année').agg({
        'Reservation_PK': 'count',
        'Jour_Férié': 'sum',
        'Mois': 'first',
        'Trimestre': 'first'
    }).reset_index().rename(columns={'Reservation_PK': 'reservations'})

    annee = '2024'
    df_weekly['Semaine_Année'] = df_weekly['Semaine_Année'].astype(int)
    df_weekly['ds'] = pd.to_datetime(annee + df_weekly['Semaine_Année'].astype(str).str.zfill(2) + '-1', format='%G%V-%u')
    df_weekly = df_weekly.sort_values('ds')
    df_weekly.set_index('ds', inplace=True)

    # ARIMA
    model = ARIMA(df_weekly['reservations'], order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)

    # SARIMA
    sarima_model = SARIMAX(df_weekly['reservations'],
                           order=(1,1,1),
                           seasonal_order=(1,1,1,52),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    sarima_fit = sarima_model.fit()
    sarima_forecast = sarima_fit.forecast(steps=12)

    forecast_dates = pd.date_range(start=df_weekly.index[-1], periods=13, freq='W')[1:]

    # Graphique
    plt.figure(figsize=(12,5))
    plt.plot(df_weekly['reservations'], label='Historique')
    plt.plot(forecast_dates, forecast, label='ARIMA(1,1,1)', linestyle='--', color='orange')
    plt.plot(forecast_dates, sarima_forecast, label='SARIMA(1,1,1)(1,1,1,52)', linestyle='-.', color='green')
    plt.title("Prévision des réservations hebdomadaires")
    plt.xlabel("Semaine")
    plt.ylabel("Réservations")
    plt.legend()
    plt.grid(True)

    # Convertir le graphique en image URI
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plot_uri = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    # Résumé simple du modèle
    summary_text = sarima_fit.summary().as_text()

    return plot_uri, summary_text
