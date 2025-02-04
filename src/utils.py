# import pandas as pd
# from sklearn.model_selection import train_test_split


# # train_path = "/Users/abhijeetthombare/ab_lib/Projects/QML/src/train.py"
# def split_data(train_path, val_path, test_size=0.2):
#     df = pd.read_csv(train_path)

#     # Split into training and validation sets
#     train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

#     # Save datasets
#     train_df.to_csv(train_path, index=False)
#     val_df.to_csv(val_path, index=False)
#     print(f"Data split completed! Validation data saved at: {val_path}")

# if __name__ == "__main__":
#     split_data("../data/train_data.csv", "../data/validation_data.csv")


# ================================= New code for vallidation dataset

# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split

# def split_data(train_path, val_path, test_size=0.2):
#     if not os.path.exists(train_path):
#         raise FileNotFoundError(f"File not found: {train_path}")

#     df = pd.read_csv(train_path)

#     if df.empty:
#         raise ValueError(f"Dataframe is empty: {train_path}")

#     # Split into training and validation sets
#     train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

#     # Save datasets
#     train_df.to_csv(train_path, index=False)
#     val_df.to_csv(val_path, index=False)

#     print(f"Data split completed! Validation data saved at: {val_path}")

# if __name__ == "__main__":
#     split_data("../data/train_data.csv", "../data/validation_data.csv")


# ================================== 2nd code

import pandas as pd
import json
import os


# Function to log model hyperparameters
def save_hyperparameters(filepath, params):
    with open(filepath, "w") as f:
        json.dump(params, f, indent=4)
    print(f"✅ Hyperparameters saved to {filepath}")


# Function to log training & evaluation metrics
def log_metrics(filepath, metrics):
    with open(filepath, "a") as f:
        f.write(metrics + "\n")
    print(f"✅ Metrics logged to {filepath}")


# Function to split data into train and validation sets
def split_data(train_path, val_path, test_size=0.2):
    df = pd.read_csv(train_path)
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"✅ Data split completed! Validation data saved at: {val_path}")
