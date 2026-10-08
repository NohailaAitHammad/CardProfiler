import mlflow
from sklearn.model_selection import  train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('classic ML')

mlflow.sklearn.autolog()

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)