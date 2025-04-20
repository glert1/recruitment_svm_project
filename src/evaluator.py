from sklearn.metrics import  accuracy_score, classification_report, confusion_matrix


def evaluate_model(y_test, y_pred):
    
    print("Model Evaluation Results:\n"
          "-----------------------------------")
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Classification Report:")
    print(report)
    print("-----------------------------------")
    