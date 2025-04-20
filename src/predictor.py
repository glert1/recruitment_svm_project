import numpy as np

def predict_candidate(model, scaler):
    
    try:
        tecrube_yili = float(input("Tecrübe Yılı (0-10): "))
        teknik_puan = float(input("Teknik Puan (0-100): "))
        
        if not (0 <= tecrube_yili <= 10 and 0 <= teknik_puan <= 100):
            print("Tecrübe Yılı must be between 0 and 10, and Teknik Puan must be between 0 and 100.")
            return
        
        user_input = np.array([[tecrube_yili, teknik_puan]])
        user_input_scaled = scaler.transform(user_input)
        
        predicton = model.predict(user_input_scaled)[0]
        if predicton == 0:
            print("Candidate is ACCEPTED for this position.")
        else:
            print("Candidate is REJECTED for this position.")
    except ValueError:
        print("Invalid input. Please enter numeric values for Tecrübe Yılı and Teknik Puan.")

