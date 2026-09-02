from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Klasyfikator irysów",
    page_icon="🌸",
)

st.title("Klasyfikator gatunków irysów")
st.write(
    "Podaj wymiary kwiatu, a model KNN przewidzi jego gatunek."
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "knn_iris.pkl"
    return joblib.load(model_path)


model = load_model()

sepal_length = st.number_input(
    "Długość działki kielicha (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1,
)

sepal_width = st.number_input(
    "Szerokość działki kielicha (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1,
)

petal_length = st.number_input(
    "Długość płatka (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1,
)

petal_width = st.number_input(
    "Szerokość płatka (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1,
)

if st.button("Przewidź gatunek", type="primary"):
    sample = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ],
    )

    prediction = model.predict(sample)[0]

    st.success(f"Przewidziany gatunek: **{prediction}**")
    st.dataframe(sample, hide_index=True)