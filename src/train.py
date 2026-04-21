# preprocessing pipeline implementation

# importing needed libraries

import pandas as pd 
import numpy as np   
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer





  # extracting numerical and categorical features

def build_pipeline(model, categorical_encoder, numeric_features, categorical_features):

    

    
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline (pluggable)
    categorical_pipeline = Pipeline([
        
        ("encoder", categorical_encoder)
    ])
    
    # Combine
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    
    # Final pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    
    return pipeline



def train_and_evaluate(X,y):

    numeric_features = X.select_dtypes(include='number').columns.tolist()
    categorical_features= X.select_dtypes(include='object').columns.tolist()


    # Build pipelines
    pipeline_lr = build_pipeline(
        LogisticRegression(max_iter=1000),
        OneHotEncoder(handle_unknown="ignore"),
        numeric_features,
        categorical_features
    )
    
    pipeline_rf = build_pipeline(
        RandomForestClassifier(),
        OrdinalEncoder(),
        numeric_features,
        categorical_features
    )
    
    # Cross-validation
    cv_lr = cross_validate(pipeline_lr, X, y, cv=5, scoring="accuracy")
    cv_rf = cross_validate(pipeline_rf, X, y, cv=5, scoring="accuracy")
    
    # Compare
    lr_score = cv_lr["test_score"].mean()
    rf_score = cv_rf["test_score"].mean()
    
    print("Logistic Regression Accuracy:", lr_score)
    print("Random Forest Accuracy:", rf_score)


    # Select best model
    if rf_score > lr_score:
        best_pipeline = pipeline_rf
        print("Best model: Random Forest")
    else:
        best_pipeline = pipeline_lr
        print("Best model: Logistic Regression")
    
    # Fit best model on FULL data
    best_pipeline.fit(X, y)
    
    return best_pipeline
    



