from sklearn.svm import SVC
import joblib
import os

def train_model(X_train, y_train,kernel= 'linear', C= 1.0, model_path='models/linear_svm_model.pkl'):
 
    model = SVC(kernel=kernel, C=C, random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
 
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    
    print(f"Scaler saved to models/scaler.pkl")
    
    return model

