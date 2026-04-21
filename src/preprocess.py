import pandas as pd
import numpy as np   

# Load data
def load_data(path):

    # read path and load it

    df = pd.read_csv(path)

    # drop Cabin

    df=df.drop(labels="Cabin",axis=1)

    # Drop Name,ticket columns

    df = df.drop(columns=["Name", "Ticket"])

    # features, targets split

    df=df.dropna(subset=['Embarked'])

    X=df.drop('Survived',axis=1)
    Y=df['Survived']

    return X,Y





   