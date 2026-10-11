def preprocessing(file_path):
      import pandas as pd
      import matplotlib.pyplot as plt
      from  data_profiling import ProfileReport

      #Chargement des données
      df = pd.read_csv(file_path)

      #Copie de df
      df_clean = df.copy()

      #Suppression des collonnes
      df_clean = df_clean.drop(columns=['CUST_ID'])
      
      #Nettoyagae minimal
      df_clean.loc[
            df_clean['CREDIT_LIMIT'].isna(),'CREDIT_LIMIT'
            ] = df_clean['CREDIT_LIMIT'].median()

      return df_clean
