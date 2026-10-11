from .classification  import *
from .preprocessing import *
from .clustering import *
from .pipeline import *

def main():
      df_clean = preprocessing("data/raw/dataset.csv")
      print('preprocessing done')

      X_scaled, scaler = clustring(df_clean)
      print('clustring done')


      df_clean= classificationn(X_scaled, scaler , df_clean)
      print('classification done')


      X_train, X_test, y_train, y_test = split_data(df_clean)
      print('split done')


      pipeline_reference(X_train, X_test, y_train, y_test)


      best_model_name, best_model, best_cv_f1 = pipeline_GS_CV(X_train, X_test, y_train, y_test, k_fold=3)

      serialisation(
      best_model,
      "models/champion_cardprofiler.joblib"
      )

      # Enregistrement du champion dans MLflow
      enregistrer_champion_mlflow(
      best_model_name,
      best_model,
      best_cv_f1
      )

      print('pipeline cv done')

main()
