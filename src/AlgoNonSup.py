import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()

df_faitAchat = pd.read_csv('Achat.csv', encoding='ISO-8859-1')

df_clients = pd.read_csv('Dim_Clients.csv', encoding='ISO-8859-1')

df_temps = pd.read_csv('Dim_Temps.csv', encoding='ISO-8859-1')

df_faitReservation = pd.read_csv('Fait_Reservations.csv', encoding='ISO-8859-1')

# 2. Préparation des données agrégées (indicateurs par client)
# -- Montant total des achats, nombre d’achats
achat_stats = df_faitAchat.groupby('Client_FK').agg(
    montant_total_achat=pd.NamedAgg(column='Prix_Total', aggfunc='sum'),
    nb_achats=pd.NamedAgg(column='AchatID_PK', aggfunc='count')
).reset_index()

# -- Statistiques de réservation
reservation_stats = df_faitReservation.groupby('ClientID_FK').agg(
    montant_total_reservation=pd.NamedAgg(column='Total_Prix', aggfunc='sum'),
    nb_reservations=pd.NamedAgg(column='Reservation_PK', aggfunc='count'),
    note_moyenne=pd.NamedAgg(column='Note_Client', aggfunc='mean')
).reset_index()

# -- Nombre de jours de voyage si Dates disponibles (on suppose que c'est la même date aller-retour, juste pour exemple)
df_faitReservation = df_faitReservation.merge(df_temps, left_on='Temps_FK', right_on='TempsID')
voyage_stats = df_faitReservation.groupby('ClientID_FK').agg(
    nb_jours_voyage=pd.NamedAgg(column='Date_Complète', aggfunc='nunique')  # approximation
).reset_index()

# 3. Fusionner les statistiques
df_final = achat_stats.merge(reservation_stats, left_on='Client_FK', right_on='ClientID_FK', how='outer')
df_final = df_final.merge(voyage_stats, on='ClientID_FK', how='left')

# 4. Remplir les NaN (0 si aucune réservation ou achat)
df_final.fillna(0, inplace=True)

# 5. Sélection des variables explicatives
features = ['montant_total_achat', 'nb_achats', 'montant_total_reservation',
            'nb_reservations', 'note_moyenne', 'nb_jours_voyage']
X = df_final[features]


# 6. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 7. KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_final['cluster'] = kmeans.fit_predict(X_scaled)

# 8. Visualisation avec PCA
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

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# ------------------- MÉTRIQUES K-MEANS -------------------
inertie_kmeans = kmeans.inertia_
silhouette_kmeans = silhouette_score(X_scaled, df_final['cluster'])
db_index_kmeans = davies_bouldin_score(X_scaled, df_final['cluster'])

print("🔷 Résultats K-Means")
print(f"Inertie : {inertie_kmeans:.2f}")
print(f"Silhouette Score : {silhouette_kmeans:.3f}")
print(f"Davies-Bouldin Index : {db_index_kmeans:.3f}")


# ------------------- DBSCAN -------------------
dbscan = DBSCAN(eps=1.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df_final['cluster_dbscan'] = dbscan_labels

# Il se peut qu'il y ait des -1 (bruit), on vérifie que le clustering est significatif
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


# ------------------- Agglomerative Clustering -------------------
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


# Sélectionner les colonnes à exporter
colonnes_export = ['Client_FK', 'montant_total_achat', 'nb_achats',
                   'montant_total_reservation', 'nb_reservations',
                   'note_moyenne', 'nb_jours_voyage', 'cluster']

# Export CSV
df_final[colonnes_export].to_csv('resultats_kmeans_clients.csv', index=False)

# Télécharger dans Google Colab
files.download('resultats_kmeans_clients.csv')


# Supprimer les clients avec clé invalide (0 ou null)
df_faitAchat = df_faitAchat[df_faitAchat['Client_FK'] > 0]
df_faitReservation = df_faitReservation[df_faitReservation['ClientID_FK'] > 0]


# Supprimer les lignes où aucun identifiant client n'est défini
df_final = df_final[df_final['Client_FK'].notna() & (df_final['Client_FK'] > 0)]


df_final['Client_FK'] = df_final['Client_FK'].astype(int)


colonnes_export = ['Client_FK', 'montant_total_achat', 'nb_achats',
                   'montant_total_reservation', 'nb_reservations',
                   'note_moyenne', 'nb_jours_voyage', 'cluster']
df_final[colonnes_export].to_csv('resultats_kmeans_clients.csv', index=False)
files.download('resultats_kmeans_clients.csv')


# Supprimer les lignes avec Client_FK ou ClientID_FK nul ou vide
df_faitAchat = df_faitAchat[df_faitAchat['Client_FK'].notna() & (df_faitAchat['Client_FK'] > 0)]
df_faitReservation = df_faitReservation[df_faitReservation['ClientID_FK'].notna() & (df_faitReservation['ClientID_FK'] > 0)]


df_faitAchat['Client_FK'] = df_faitAchat['Client_FK'].astype(int)
df_faitReservation['ClientID_FK'] = df_faitReservation['ClientID_FK'].astype(int)


achat_stats = df_faitAchat.groupby('Client_FK').agg(
    montant_total_achat=('Prix_Total', 'sum'),
    nb_achats=('AchatID_PK', 'count')
).reset_index()

reservation_stats = df_faitReservation.groupby('ClientID_FK').agg(
    montant_total_reservation=('Total_Prix', 'sum'),
    nb_reservations=('Reservation_PK', 'count'),
    note_moyenne=('Note_Client', 'mean')
).reset_index()


# Fusion achat + réservation
df_final = achat_stats.merge(reservation_stats, left_on='Client_FK', right_on='ClientID_FK', how='left')

# Fusion avec nb jours de voyage
df_faitReservation = df_faitReservation.merge(df_temps, left_on='Temps_FK', right_on='TempsID')
voyage_stats = df_faitReservation.groupby('ClientID_FK').agg(
    nb_jours_voyage=('Date_Complète', 'nunique')
).reset_index()

df_final = df_final.merge(voyage_stats, on='ClientID_FK', how='left')


df_final.fillna({
    'montant_total_reservation': 0,
    'nb_reservations': 0,
    'note_moyenne': 0,
    'nb_jours_voyage': 0
}, inplace=True)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Sélection des variables explicatives
features = ['montant_total_achat', 'nb_achats', 'montant_total_reservation',
            'nb_reservations', 'note_moyenne', 'nb_jours_voyage']

X = df_final[features]

# 2. Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Clustering avec KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_final['cluster'] = kmeans.fit_predict(X_scaled)


for col in ['montant_total_reservation', 'nb_reservations', 'note_moyenne', 'nb_jours_voyage']:
    moyenne = df_final[col].mean()
    df_final[col].fillna(moyenne, inplace=True)


# Optionnel : ne garder que les colonnes utiles
colonnes_export = ['Client_FK'] + features + ['cluster']
df_final[colonnes_export].to_csv('resultats_kmeans_clients.csv', index=False)

