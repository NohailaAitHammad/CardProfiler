def clustring(df_clean):
      import pandas as pd
      import numpy as np
      from sklearn.preprocessing import StandardScaler



      #Préparation du clustring
      df_prepare_clustering = df_clean.copy()

      #Transformation Logarithmique
      X_log = np.log1p(df_prepare_clustering)

      #Standardisation
      scaler = StandardScaler()

      X_scaled = scaler.fit_transform(X_log)

      return X_scaled, scaler