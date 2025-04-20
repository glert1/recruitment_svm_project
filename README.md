# Recruitment SVM Predictor

This project is a machine learning application that uses Support Vector Machine (SVM) to predict whether a candidate should be accepted or rejected for a position based on their experience and technical score.

## Features

- Generates synthetic candidate data
- Trains an SVM model with customizable parameters
- Visualizes decision boundaries
- Provides a FastAPI web service for predictions
- Interactive command-line interface for predictions
- Model evaluation metrics

## Project Structure

```
recruitment_svm_project/
├── src/
│   ├── data_generator.py    # Generates synthetic candidate data
│   ├── preprocessing.py     # Data preprocessing and scaling
│   ├── trainer.py          # Model training functionality
│   ├── predictor.py        # Prediction interface
│   ├── evaluator.py        # Model evaluation metrics
│   ├── visualizer.py       # Visualization tools
│   ├── fastapi_service.py  # FastAPI web service
│   └── __init__.py
├── models/                  # Directory for saved models
├── plots/                   # Directory for saved plots
├── data/                    # Directory for generated data
├── app.py                   # Main application script
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd recruitment_svm_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
or you can just run the install_requirements.py

## Usage

### Training the Model

Run the main application to train the model and generate visualizations:
```bash
python app.py
```

This will:
- Generate synthetic candidate data
- Train the SVM model
- Save the model and scaler
- Generate and save decision boundary plots
- Evaluate the model performance

### Making Predictions


#### FastAPI Web Service
Start the FastAPI service:
```bash
uvicorn src.fastapi_service:app --reload --port 8002
```

Access the Swagger UI documentation at:
```
http://localhost:8002/docs
```

Make predictions using the `/predict` endpoint with the following JSON format:
```json
{
    "tecrube_yili": 5.0,
    "teknik_puan": 75.0
}
```

## API Documentation

The FastAPI service provides the following endpoints:

- `POST /predict`: Make a prediction for a candidate
  - Input: JSON with `tecrube_yili` (0-10) and `teknik_puan` (0-100)
  - Output: Prediction result (0 for ACCEPTED, 1 for REJECTED)

## Model Parameters

The SVM model uses the following default parameters:
- Kernel: linear
- C: 1.0
- Random State: 42

These can be modified in the `trainer.py` file.

## Data Generation

The synthetic data is generated with the following characteristics:
- Experience years (tecrube_yili): 0-10
- Technical score (teknik_puan): 0-100
- Label: 0 (ACCEPTED) or 1 (REJECTED)

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- joblib
- fastapi
- uvicorn
- pydantic

