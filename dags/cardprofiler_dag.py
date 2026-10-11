from datetime import datetime, timedelta
from airflow.decorators import dag, task
from src.main import main as run_cardprofiler

from src.classification  import *
from src.preprocessing import *
from src.clustering import *
from src.pipeline import *


@dag(
      dag_id="CardProfiler",
      start_date=datetime(2026, 10, 11),
      schedule="@daily",
)

def CardProfiler():
      @task(
        retries=3,
        retry_delay=timedelta(minutes=1),
    )
      def preprocessing_task(file_path):
            return preprocessing(file_path)

      @task(
              retries=3,
              retry_delay=timedelta(minutes=1),
          )
      def clustring_task(df_clean):
            return clustring(df_clean)

      @task(
            retries=3,
            retry_delay=timedelta(minutes=1),
            )
      def classification_task(X_scaled, scaler , df_clean):
            return classificationn(X_scaled, scaler , df_clean)

      @task(
            retries=3,
            retry_delay=timedelta(minutes=1),
            )
      def spliting_task(df_clean):
            return split_data(df_clean)

      @task(
                  retries=3,
                  retry_delay=timedelta(minutes=1),
                  )
      def pipeline_reference_task(X_train, X_test, y_train, y_test):
            return pipeline_reference(X_train, X_test, y_train, y_test)

      
      @task(
            retries=3,
            retry_delay=timedelta(minutes=1),
            )
      def pipeline_GS_CV_task(X_train, X_test, y_train, y_test, k_fold):
            return pipeline_GS_CV(X_train, X_test, y_train, y_test, k_fold)


      @task(
            retries=3,
            retry_delay=timedelta(minutes=1),
            )
      def serialisation_task(best_model, path):
            return serialisation(
      best_model,
      path
      )

      @task(
                  retries=3,
                  retry_delay=timedelta(minutes=1),
                  )
      def enregistrer_champion_mlflow_task(best_model_name,
                        best_model,
                        best_cv_f1):
            return enregistrer_champion_mlflow(
                  best_model_name,
                  best_model,
                  best_cv_f1
                  )

CardProfiler()