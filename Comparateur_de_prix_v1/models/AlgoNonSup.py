# /models/algo_non_sup.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering

def load_data():
    # Remplacez ici les chemins des fichiers locaux ou d'autres méthodes pour charger les données
    df_faitAchat = pd.read_csv('Achat.csv', encoding='ISO-8859-1')
    df_clients = pd.read_csv('Dim_Clients.csv', encoding='ISO-8859-1')
    df_temps = pd.read_csv('Dim_Temps.csv', encoding='ISO-8859-1')
    df_faitReservation = pd.read_csv('Fait_Reservations.csv', encoding='ISO-8859-1')
    
    return df_faitAchat, df_clients, df_temps, df_faitReservation

def prepare_data(df_faitAchat, df_faitReservation, df_temps):
    # Préparer les données agrégées par client (achats, réservations, voyages)
    achat_stats = df_faitAchat.groupby('Client_FK').agg(
        montant_total_achat=('Prix_Total', 'sum'),
        nb_achats=('AchatID_PK', 'count')
    ).reset_index()

    reservation_stats = df_faitReservation.groupby('ClientID_FK').agg(
        montant_total_reservation=('Total_Prix', 'sum'),
        nb_reservations=('Reservation_PK', 'count'),
        note_moyenne=('Note_Client', 'mean')
    ).reset_index()

    df_faitReservation = df_faitReservation.merge(df_temps, left_on='Temps_FK', right_on='TempsID')
    voyage_stats = df_faitReservation.groupby('ClientID_FK').agg(
        nb_jours_voyage=('Date_Complète', 'nunique')
    ).reset_index()

    # Fusionner les statistiques
    df_final = achat_stats.merge(reservation_stats, left_on='Client_FK', right_on='ClientID_FK', how='outer')
    df_final = df_final.merge(voyage_stats, on='ClientID_FK', how='left')

    df_final.fillna(0, inplace=True)  # Remplir les valeurs manquantes avec 0
    return df_final

def run_clustering(df_final):
    # Sélection des variables explicatives
    features = ['montant_total_achat', 'nb_achats', 'montant_total_reservation',
                'nb_reservations', 'note_moyenne', 'nb_jours_voyage']
    X = df_final[features]

    # Standardisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_final['cluster'] = kmeans.fit_predict(X_scaled)

    return df_final, X_scaled, kmeans

def visualize_clusters(df_final, X_scaled):
    # Visualisation avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_final['pca1'] = X_pca[:, 0]
    df_final['pca2'] = X_pca[:, 1]

    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df_final, x='pca1', y='pca2', hue='cluster', palette='Set2')
    plt.title("Clustering des clients (PCA)")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend(title="Cluster")
    plt.show()

def calculate_metrics(X_scaled, df_final, kmeans):
    # Métriques pour KMeans
    silhouette_kmeans = silhouette_score(X_scaled, df_final['cluster'])
    db_index_kmeans = davies_bouldin_score(X_scaled, df_final['cluster'])

    print("🔷 Résultats K-Means")
    print(f"Silhouette Score : {silhouette_kmeans:.3f}")
    print(f"Davies-Bouldin Index : {db_index_kmeans:.3f}")

    # DBSCAN
    dbscan = DBSCAN(eps=1.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df_final['cluster_dbscan'] = dbscan_labels

    if len(set(dbscan_labels)) > 1:
        silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)
        db_index_dbscan = davies_bouldin_score(X_scaled, dbscan_labels)
    else:
        silhouette_dbscan = None
        db_index_dbscan = None

    print("\n🔶 Résultats DBSCAN")
    print(f"Nombre de clusters (hors bruit) : {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")
    print(f"Silhouette Score : {silhouette_dbscan:.3f}" if silhouette_dbscan else "Silhouette Score : non applicable")
    print(f"Davies-Bouldin Index : {db_index_dbscan:.3f}" if db_index_dbscan else "Davies-Bouldin Index : non applicable")

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=3)
    agglo_labels = agglo.fit_predict(X_scaled)
    df_final['cluster_agglo'] = agglo_labels

    silhouette_agglo = silhouette_score(X_scaled, agglo_labels)
    db_index_agglo = davies_bouldin_score(X_scaled, agglo_labels)

    print("\n🔸 Résultats Agglomerative Clustering")
    print(f"Silhouette Score : {silhouette_agglo:.3f}")
    print(f"Davies-Bouldin Index : {db_index_agglo:.3f}")

    results_df = pd.DataFrame({
        'Algorithme': ['KMeans', 'DBSCAN', 'Agglomerative'],
        'Silhouette Score': [silhouette_kmeans, silhouette_dbscan if silhouette_dbscan else None, silhouette_agglo],
        'Davies-Bouldin Index': [db_index_kmeans, db_index_dbscan if db_index_dbscan else None, db_index_agglo]
    })

    print("\n📋 Comparaison des algorithmes :")
    print(results_df)

def export_results(df_final):
    # Sélectionner les colonnes à exporter
    colonnes_export = ['Client_FK', 'montant_total_achat', 'nb_achats',
                       'montant_total_reservation', 'nb_reservations',
                       'note_moyenne', 'nb_jours_voyage', 'cluster']
    
    df_final[colonnes_export].to_csv('resultats_kmeans_clients.csv', index=False)
    print("Fichier exporté sous 'resultats_kmeans_clients.csv'")

# Main function to run the algorithm
def run_algo_non_sup():
    df_faitAchat, df_clients, df_temps, df_faitReservation = load_data()
    df_final = prepare_data(df_faitAchat, df_faitReservation, df_temps)
    df_final, X_scaled, kmeans = run_clustering(df_final)
    visualize_clusters(df_final, X_scaled)
    calculate_metrics(X_scaled, df_final, kmeans)
    export_results(df_final)