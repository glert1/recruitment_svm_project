from src.data_generator import generate_candidate_data
from src.preprocessing import prepare_data
from src.predictor import predict_candidate
from src.visualizer import plot_decision_boundary
from src.trainer import train_model
from src.evaluator import evaluate_model
import numpy as np
import joblib
import os

def main():
    
    print("Generating candidate data...")
    df = generate_candidate_data(n_samples=500)
    
    print("Preparing data...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    print("Training the SVM model...")
    model = train_model(X_train, y_train)
    joblib.dump(scaler, 'models/scaler.pkl')  
    
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    print('Plotting decision boundary...')
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    os.makedirs('plots', exist_ok=True)
    
    plot_decision_boundary(
        model, 
        X_combined, 
        y_combined, 
        scaler=scaler,
        save_path='plots/linear_decision_boundary.png'
    )
    
    print("Predicting new candidates...")
    predict_candidate(model, scaler)
  
    
    
if __name__ == "__main__":
    main()