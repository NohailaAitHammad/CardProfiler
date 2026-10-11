def split_data(df_clean):
      from sklearn.model_selection import train_test_split
      columns_to_remove = [
      'target',
      'cluster_kmeans',
      'cluster_dbscan',
      'est_atypique_dbscan',
      'cluster_final'
      ]

      X = df_clean[df_clean.columns.difference(other=columns_to_remove)]

      y = df_clean['target']

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

      return X_train, X_test, y_train, y_test

def pipeline_reference(X_train, X_test, y_train, y_test):
      import  pandas as pd
      import numpy as np
      from sklearn.pipeline import Pipeline
      from sklearn.linear_model import LogisticRegression
      from sklearn.svm import SVC
      from sklearn.preprocessing import StandardScaler
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
      import mlflow
      import mlflow.sklearn
      from pathlib import Path

      models_references = {
     'LogisticRegression' : (
            LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            ),
      'DecisionTreeClassifier' : (
                  DecisionTreeClassifier(random_state=42)
                 ),
      'RandomForestClassifier' : (
            RandomForestClassifier(random_state=42)
           ),
      'SVC' : (
                 SVC(kernel='linear', C=1)
           )
      }

      results = []

      for models_name, (model) in models_references.items():

            pipeline = Pipeline(steps=[
                        ('scaler', StandardScaler()),
                        ('classificator', model)
                  ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            log_classification_run(run_name=f'reference_{models_name}', pipeline=pipeline, X_test=X_test,y_test=y_test,   y_pred=y_pred, params={
                  "model_type" : models_name,
                  "scaler_type" : "StandardScaler",
                  "test_size" : 0.2,
                  "random_state" : 42,
                  **{
                        f"moldel_{key}": value 
                        for key, value in model.get_params().items()
                  }
            })
            results.append({
                  "Model": models_name,
                  "Accuracy": accuracy_score(y_test, y_pred),
                  "Precision": precision_score(y_test, y_pred, average='macro'),
                  "Recall": recall_score(y_test, y_pred, average='macro'),
                  "F1_Score": f1_score(y_test, y_pred, average='macro'),
            })
            

      result_df = pd.DataFrame(results).sort_values(by='F1_Score', ascending=False, ignore_index=True)
  
def pipeline_GS_CV(X_train, X_test, y_train, y_test, k_fold):
      import pandas as pd
      import numpy as np
      from sklearn.pipeline import Pipeline
      from sklearn.linear_model import LogisticRegression
      from sklearn.svm import SVC
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
      from scipy.stats import randint
      from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
      from sklearn.preprocessing import StandardScaler

      # models_GSCV = {
      #      'LogisticRegression' : (
      #                  LogisticRegression(random_state=42), 
      #                  {
      #                   'classificator__C': [0.01, 0.1, 1, 10],
      #                   'classificator__l1_ratio': [0.0],
      #                   'classificator__solver': ['saga'],
      #                   'classificator__max_iter': [100, 500],
      #                  }
      #            ),
      #       'DecisionTreeClassifier' : (
      #                              DecisionTreeClassifier( random_state=42),
      #                              {
      #                               'classificator__max_depth' : range(1, 10, 1),
      #                               'classificator__min_samples_leaf' : range(1, 20, 2),
      #                               'classificator__min_samples_split' : range(2, 20, 2),
      #                               'classificator__criterion' : ['entropy', 'gini']   
      #                              }
      #             ),
      #       'RandomForestClassifier' : (
      #                  RandomForestClassifier( random_state=42, n_jobs=1),
      #                  {
      #                   'classificator__max_depth' : range(1, 10, 1),
      #                   'classificator__min_samples_leaf' : range(1, 20, 2),
      #                   'classificator__min_samples_split' : range(2, 20, 2),
      #                   'classificator__criterion' : ['entropy', 'gini']   
      #                  }
      #            ),
      #       'SVC':(
      #                  SVC(random_state=42),
      #                  {
      #                     'classificator__C': [0.1, 1, 10],
      #                     'classificator__kernel': ['linear', 'rbf'],
      #                     'classificator__gamma': ['scale', 'auto']   
      #                  }
      #             )
      # }

      models_GSCV = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=2000,
                l1_ratio=0.0,
                random_state=42
            ),
            {
                "classificator__C": [0.1, 1, 10]
            },
            3
        ),

        "DecisionTreeClassifier": (
            DecisionTreeClassifier(random_state=42),
            {
                "classificator__max_depth": [None, 3, 5, 8, 12],
                "classificator__min_samples_leaf": [1, 2, 5, 10],
                "classificator__min_samples_split": [2, 5, 10, 20],
                "classificator__criterion": ["gini", "entropy"]
            },
            12
        ),

        "RandomForestClassifier": (
            RandomForestClassifier(
                random_state=42,
                n_jobs=1
            ),
            {
                "classificator__n_estimators": [100, 150],
                "classificator__max_depth": [None, 5, 10, 15],
                "classificator__min_samples_leaf": [1, 2, 4, 8],
                "classificator__min_samples_split": [2, 5, 10],
                "classificator__criterion": ["gini", "entropy"]
            },
            10
        ),

        "SVC": (
            SVC(random_state=42),
            {
                "classificator__C": [0.1, 1, 10],
                "classificator__kernel": ["linear", "rbf"],
                "classificator__gamma": ["scale", "auto"]
            },
            8
        )
    }
      results_cv = []
      best_models = {}

      cv_strategy = StratifiedKFold(
        n_splits=k_fold,
        shuffle=True,
        random_state=42
    )
      
      for models_name, (model, params, n_iter) in models_GSCV.items():
            print(f"\nRecherche en cours : {models_name}")
            pipeline = Pipeline(steps=[
                        ('scaler', StandardScaler()),
                        ('classificator', model)
                  ])
            # gs = GridSearchCV(estimator=pipeline, param_grid=params, cv=k_fold, n_jobs=-1, scoring='f1_macro')
            gs=  RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
            verbose=1,
            refit=True
        )

            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_score = gs.best_score_

            y_pred = best_model.predict(X_test)

            log_classification_run(
                  run_name=f"randomizedsearch_{models_name}",
                  pipeline=best_model,
                  X_test=X_test,
                  y_test=y_test,
                  y_pred=y_pred,
                  params={
                        "model_type": models_name,
                        "search_method": "RandomizedSearchCV",
                        "n_iter": n_iter,
                        "cv_folds": k_fold,
                        "scoring": "f1_macro",
                        **{
                              f"best_{key}": value
                              for key, value in gs.best_params_.items()
                        }
                  },
                  cv_f1=gs.best_score_
                  )

            
            results_cv.append({
            "Model": models_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1_Score": f1_score(y_test, y_pred, average='macro'),
            "CV_F1": best_score
      })

            print(f"\n===== {models_name} =====")
            print("Meilleurs paramètres :", gs.best_params_)

            best_models[models_name] = gs.best_estimator_

            df_results = pd.DataFrame(results_cv)

            df_results = df_results.sort_values(by='CV_F1', ascending=False).reset_index(drop=True)

      print("\nComparaison des modèles :")
      print(df_results)

      # Selectionner le champion
      best_model_name = df_results.loc[0, 'Model']

      best_cv_f1 = df_results.loc[0 ,'F1_Score']

      best_model = best_models[best_model_name]

      print("\n===== CHAMPION =====")

      print('Model : ', best_model_name)

      print('CV score : ', best_cv_f1)

      y_pred_best = best_model.predict(X_test)

      return best_model_name, best_model, best_cv_f1

def enregistrer_champion_mlflow(best_model_name, best_model, best_cv_f1):
      import mlflow
      import mlflow.sklearn
      from mlflow import MlflowClient

      mlflow.set_tracking_uri('http://127.0.0.1:5000')
      mlflow.set_experiment('Classification_Clients')

      model_registry_name = "CardProfiler_Classifier"


      #enregistrer me pipeline champion dans un nouveau run

      with mlflow.start_run(run_name=f"champion_{best_model_name}") as run :
            mlflow.log_param('champion_model', best_model_name)
            mlflow.log_metric('cv_f1_macro', float(best_cv_f1))

            mlflow.sklearn.log_model(
                  sk_model=best_model,
                  name="model",
                  serialization_format='cloudpickle'
            )

            champion_run_id = run.info.run_id

      # Enregistrer le model dans le model Registry

      model_uri = f"runs:/{champion_run_id}/model"

      model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name
      )

      # Attribuer l'alisa champion a cette version

      client = MlflowClient()

      client.set_registered_model_alias(
            name=model_registry_name,
            alias='champion',
            version=model_version.version
      )

      print("\nChampion enregistrer avec success")
      print("Model : ", best_model_name)
      print('CV_F1', best_cv_f1)
      print('Version MLflow', model_version.version)
      print('Alisas : champion')
      
def serialisation(best_model, path):
      import joblib

      joblib.dump(best_model, path)
      print('Model Suavgarder avec success')

def log_classification_run(run_name, pipeline, X_test, y_test, y_pred, params, cv_f1=None):
      import matplotlib.pyplot as plt
      import mlflow
      import mlflow.sklearn
      from pathlib import Path
      from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,ConfusionMatrixDisplay 

      # 1 parametre de l'experience
      mlflow.set_tracking_uri('http://127.0.0.1:5000')
      mlflow.set_experiment('Classification_Clients')

      #2 creer un run
      with mlflow.start_run(run_name=run_name):
            # 3 enregistrer les parameters
            mlflow.log_params({
            key: str(value) if isinstance(value, (dict, list, tuple))
            else value
            for key, value in params.items()
            })

            # calculer et enregistrer les metriques
            metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_test, y_pred, average="macro", zero_division=0
            )
        }

            if cv_f1 is not None:
                  metrics['cv_f1_macro'] = float(cv_f1)

            mlflow.log_metrics(metrics)

            # 5 enregistrer le pipeline complet
            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path='model',
            serialization_format="cloudpickle")

            # 6 enregistrer le rapport de classification

            Path('mlflow_artifacts').mkdir(exist_ok=True)

            repport_path = Path("mlflow_artifacts") / 'classification_report.txt'

            repport_path.write_text(
                  classification_report(y_test, y_pred, zero_division=0),
                  encoding='utf-8'
            )

            mlflow.log_artifact(str(repport_path))

            # 7 Enregistrer la matrice de confusion

            fig, ax = plt.subplots(figsize=(7,6))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, values_format='d')

            ax.set_title(f'Matrice de confusion - {run_name}')

            cm_path = Path('mlflow_artifacts') / 'confusion_matrix.png'
            fig.savefig(cm_path, bbox_inches='tight')
            plt.close(fig)

            mlflow.log_artifact(str(cm_path))

            print(f'Run enregistrer  : {run_name}')
            print(metrics)