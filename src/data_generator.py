import numpy as np
import pandas as pd
import os


def generate_candidate_data(n_samples=500, seed=42):
    
    np.random.seed(seed)
    tecrube_yili = np.round(np.random.uniform(0, 10, size=n_samples), 2)
    teknik_puan = np.round(np.random.uniform(0, 100, size=n_samples), 2)

    label = np.where((tecrube_yili<2) & (teknik_puan<60), 1, 0)


    df = pd.DataFrame({
        'tecrube_yili': tecrube_yili,
        'teknik_puan': teknik_puan,
        "label": label
    })

    return df


if __name__ == "__main__":
    df = generate_candidate_data(n_samples=500)
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "generate_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated data saved to {output_path}")
    print(df.head())



