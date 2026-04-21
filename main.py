from src.preprocess import load_data
from src.train import train_and_evaluate
import pandas as pd

import pickle
import os


# Loading training data

def main():

 x,y=load_data("data/raw/train.csv")

 test_df = pd.read_csv("data/raw/test.csv")
 test_df = test_df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

 best_pipeline=train_and_evaluate(x,y)

 predictions = best_pipeline.predict(test_df)

 print("Test predictions shape:", predictions.shape)


 submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
 })

 submission.to_csv("submission.csv", index=False)

  # Save model
 os.makedirs("model", exist_ok=True) 
 with open("model/best_model.pkl", "wb") as f:
   pickle.dump(best_pipeline, f)

 print("Model and submission saved successfully!")



if __name__ == "__main__":
    main()