import joblib
import pandas as pd

model = joblib.load("knn_iris.pkl")

sample = pd.DataFrame(
    [[6.0, 3.0, 4.0, 2.0]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ],
)

prediction = model.predict(sample)

print(prediction)