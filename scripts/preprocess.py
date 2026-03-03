import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_and_split(path='data/heart.csv'):
    df=pd.read_csv(path)
    print(f'Data is loaded successfully with shape: {df.shape}')

    #Handling missing values
    missing_count=df.isnull().sum().sum()
    if missing_count>0:
        df.dropna(inplace=True)
        print(f"Dropped rows with missing values:{missing_count}")
    else:
        print("\n No missing values found")
    
    #Remove duplicates rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count>0:
        df.drop_duplicates(inplace=True)
        print(f"Dropped duplicate rows:{duplicate_count}")
    else:
        print("\n No duplicates found")
    
    #Encoding categorical columns (if any)
    categorical_cols= df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        df=pd.get_dummies(df,columns=categorical_cols,drop_first=True) #get_dummies coverts categorical into numerical 
        print(f"Encoded categorical columns:{categorical_cols}")
    else:
        print("No categorical columns to encode")
    
    if 'HeartDisease' not in df.columns:
        raise ValueError('HeartDisease column not found in Dataset')
    
    X=df.drop('HeartDisease',axis=1)
    y=df['HeartDisease']

    numeric_cols=X.select_dtypes(include=['int64','Float64']).columns
    non_numeric_cols = X.columns.difference(numeric_cols)

    scaler = StandardScaler()
    X_scaled_numeric = scaler.fit_transform(X[numeric_cols])

    # Convert scaled numeric features back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled_numeric,columns=numeric_cols,index=X.index)
    
    # 🧱 Combine scaled numeric and unscaled non-numeric features
    X_processed =pd.concat([X_scaled_df,X[non_numeric_cols]],axis=1)

    # 💾 Save the scaler
    os.makedirs('models',exist_ok=True)
    with open('models/scaler.pk1','wb') as f:
        pickle.dump(scaler,f)
    
#feature selection using RandomForest
    rf= RandomForestClassifier(random_state=42)
    rf.fit(X_processed,y)
     
    selector = SelectFromModel(rf,threshold='median',prefit=True)
    X_selected = selector.transform(X_processed)

    original_features = X_processed.columns.tolist()
    selected_features = np.array(original_features)[selector.get_support()].tolist()
    print(f"Selected {len(selected_features)} important features out of {len(original_features)}")

    #saving selected features
    with open('models/selected_features.txt','w') as f:
        for feats in selected_features:
            f.write(f"{feats} \n")

    X_train,X_test,y_train,y_test = train_test_split(X_selected,y,test_size=0.2, stratify=y,random_state=42)

    return X_train,X_test,y_train,y_test,selected_features,scaler