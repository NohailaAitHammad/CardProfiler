def classificationn(X_scaled, scaler, df_clean):
      import pandas as pd
      import numpy as np
      from sklearn.decomposition import PCA
      from classification import search_best_dbscan, search_best_k_pca, fit_final_dbscan_clustering, fit_final_kmeans_clustering


      #Search  best K_PCA
      best_k, best_n_componenet = search_best_k_pca(X_scaled)

      #Entrainement PCA et Kmeans
      cluster_kmeans = fit_final_kmeans_clustering(best_n_componenet, best_k, X_scaled)

      df_clean["cluster_kmeans"] = cluster_kmeans

      results_dbscan_df, candidates = search_best_dbscan(X_scaled)

      cluster_dbscan, pca_dbscan_final, dbscan = fit_final_dbscan_clustering(X_scaled,candidates.iloc[0]['n_components'], candidates.iloc[0]['eps'], candidates.iloc[0]['min_samples'])

      df_clean["cluster_dbscan"] = cluster_dbscan

      df_clean["est_atypique_dbscan"] = (
      df_clean["cluster_dbscan"] == -1
      )

      df_clean["cluster_final"] = df_clean["cluster_kmeans"]

      cluster_names = {
            0: "Clients Premium",
            1: "Clients à Risque",
            2: "Clients à Faible Activité"
      }

      df_clean['target'] = df_clean['cluster_final'].map(cluster_names)

      return df_clean

def search_best_k_pca(X_scaled):
      import pandas as pd
      import numpy as np
      from sklearn.decomposition import PCA
      from sklearn.cluster import KMeans
      from sklearn.metrics import silhouette_score

      pca = PCA(n_components=7)

      X_pca = pca.fit_transform(X_scaled)

      variance_expliquee =pca.explained_variance_ratio_

      variance_cumule =np.cumsum(pca.explained_variance_ratio_)

      best_n_componenet = np.argmax(np.cumsum(variance_expliquee) > 0.8) + 1

      #print(f' le best n_component : {best_n_componenet} pour une variance cumulée > 80%' )

      n_composants_list = range(2,6)
      k_list = range(2,8)

      resluts_kmeans = []

      for n_component in n_composants_list:
            
      #PCA AVEC LE NOMBRE DE COMPOSANTS ACTUEL
            pca = PCA(n_components=n_component)

            X_pca = pca.fit_transform(X_scaled)

            for k in k_list:
                  #CREER LE MODEL K_MEANS
                  kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                  #ENTRAINER K_MEANS
                  labels = kmeans.fit_predict(X_pca)
                  #RECUPERER INERTIA
                  inertia  = kmeans.inertia_
                  #CALCUL SILHOUETTE
                  silhouette = silhouette_score(X_pca,labels )

                  #SAUVGARDE DES RESULTATS
                  resluts_kmeans.append({
                        "n_components" : n_component,
                        "k" : k,
                        "inertia" : inertia,
                        "silhouette":silhouette
                  })

      results_kmeans_df = pd.DataFrame(resluts_kmeans)

      results_kmeans_df = results_kmeans_df.sort_values(
            by="silhouette",
            ascending=False
            )
      best_kmeans = results_kmeans_df.iloc[0]

      best_k = int(best_kmeans["k"])
      best_n_components = int(best_kmeans["n_components"])

      #print("Meilleur K :", best_k)
      #print("Meilleur nombre de composantes PCA :", best_n_components)
      #print("Meilleur silhouette :", best_kmeans["silhouette"])

      return best_k, best_n_componenet

def search_best_dbscan(
    X_scaled,
    eps_list=(0.3, 0.5, 0.7, 1.0),
    min_samples_list=(3, 5, 10),
    n_components_list=range(2, 6),
    max_noise_percent=5,
    min_cluster_percent=1
):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    results = []
    n_samples = len(X_scaled)

    for n_components in n_components_list:

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        for eps in eps_list:
            for min_samples in min_samples_list:

                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                )
                labels = model.fit_predict(X_pca)

                # Compter les clients dans chaque cluster
                unique_labels, counts = np.unique(
                    labels[labels != -1],
                    return_counts=True
                )

                # Identifier les clusters significatifs
                min_size = n_samples * min_cluster_percent / 100

                significant_clusters = unique_labels[
                    counts >= min_size
                ]

                # Garder seulement les clients des clusters significatifs
                valid_mask = np.isin(
                    labels, significant_clusters
                )

                X_valid = X_pca[valid_mask]
                labels_valid = labels[valid_mask]

                n_clusters = len(significant_clusters)

                # Bruit + clients appartenant aux petits clusters
                noise_percent = (
                    (~valid_mask).sum() / n_samples * 100
                )

                silhouette = np.nan

                if (
                    n_clusters >= 2
                    and len(X_valid) > n_clusters
                ):
                    silhouette = silhouette_score(
                        X_valid,
                        labels_valid
                    )

                valid_segmentation = (
                    n_clusters >= 2
                    and noise_percent <= max_noise_percent
                    and not np.isnan(silhouette)
                    and silhouette > 0
                )

                results.append({
                    "n_components": n_components,
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "noise_percent": noise_percent,
                    "silhouette": silhouette,
                    "valid_segmentation": valid_segmentation
                })

    results_df = pd.DataFrame(results)

    candidates = results_df[
        results_df["valid_segmentation"]
    ].sort_values(
        "silhouette",
        ascending=False
    ).reset_index(drop=True)

#     if candidates.empty:
#         print(
#             "Aucun candidat valide. "
#             "Compare les résultats DBSCAN à K-Means."
#         )
#     else:
#         print("Meilleur candidat DBSCAN :")
#         print(candidates.iloc[0])

    return results_df, candidates
      
def fit_final_kmeans_clustering(n_components, n_clusters, X_scaled):
      import pandas as pd
      import numpy as np
      from sklearn.decomposition import PCA
      from sklearn.cluster import KMeans

      pca_final = PCA(n_components=n_components)

      X_pca_final = pca_final.fit_transform(X_scaled)

      kmeans_final = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
            )

      cluster_kmeans = kmeans_final.fit_predict(X_pca_final)
      return cluster_kmeans
 
def fit_final_dbscan_clustering( X_scaled ,n_components, eps, min_samples):

      import pandas as pd
      import numpy as np
      from sklearn.decomposition import PCA
      from sklearn.cluster import DBSCAN

      pca_dbscan_final = PCA(n_components=n_components)

      X_pca_dbscan_final = pca_dbscan_final.fit_transform(X_scaled)

      dbscan = DBSCAN(eps=eps, min_samples=min_samples)

      cluster_dbscan = dbscan.fit_predict(X_pca_dbscan_final)

      return cluster_dbscan, pca_dbscan_final, dbscan