import os
import pickle

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def train_models(X_train, y_train,model_dir='models'):
    os.makedirs(model_dir,exist_ok=True)

    smote=SMOTE(random_state=42) 
    X_train_bal,y_train_bal = smote.fit_resample(X_train,y_train)

    models={
        'XGBoost':{
            "model": XGBClassifier(random_state=42,eval_metric='logloss'),
            "params" :{
                'n_estimators':[100],
                'max_depth':[3,5],
                'learning_rate':[0.05],
                'subsample' : [0.8]
            }
        },
        'RandomForest':{
            'model':RandomForestClassifier(random_state=42),
            'params':{
                'n_estimators':[100],
                'max_depth':[5],
                'min_samples_split':[5]
            }
        },
        'DecisionTree':{
            'model':DecisionTreeClassifier(random_state=42),
            'params':{
                'max_depth': [3,5],
                'min_samples_split':[2,4]
            }
        }
    }

    for name,cfg in models.items():
        print(f"\n' Training {name}")
        with mlflow.start_run(run_name=f'{name}_training'):
            grid = GridSearchCV(cfg['model'],cfg['params'],cv=3,scoring='accuracy',n_jobs=-1)
            grid.fit(X_train_bal,y_train_bal)
            best_model=grid.best_estimator_

            mlflow.log_params(grid.best_params_)
            input_example = X_train_bal[:1]
            signature=infer_signature(X_train_bal,best_model.predict(X_train_bal))
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path=f"{name}_model",
                input_example=input_example,
                signature=signature,
                registered_model_name=f"HeartDisease_{name}"
            )

            model_path = os.path.join(model_dir,f"{name.lower()}_model.pk1")
            with open(model_path,'wb') as f:
                pickle.dump(best_model,f)

            print(f"{name} saved to {model_path} with best params: {grid.best_params_}") 