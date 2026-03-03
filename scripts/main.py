from data_loader import load_data
from preprocess import preprocess_and_split
from train import train_models
from evaluate_models import evaluate_models
import mlflow

if __name__ == "__main__":

    mlflow.set_experiment("HeartDiseasePrediction")
    
    df = load_data()
    print("\n First 5 rows:")
    print(df.head())
    
    X_train, X_test, y_train, y_test, features, scaler = preprocess_and_split()
    print("\n Final split")
    print(f"🔹 Training Samples: {X_train.shape[0]}")
    print(f"🔹 Testing Samples: {X_test.shape[0]}")
    print(f"📌 Features Used: {features}")

    # Train all models 
    print("🏋️ Training models...")
    train_models(X_train, y_train)

    # evaluate models
    print("Evaluating models")
    results = evaluate_models(X_test,y_test)
    
    print("Pipeline completed")