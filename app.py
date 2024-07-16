# app.py

import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

def predict_disease_from_symptom(symptom_list):
    symptoms = { ... }  # Same dictionary from previous example

    for s in symptom_list:
        symptoms[s] = 1

    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))

    clf = load(str("./saved_model/random_forest.joblib"))
    result = clf.predict(df_test)

    del df_test

    return f"{result[0]}"

iface = gr.Interface(
    predict_disease_from_symptom,
    [
        gr.inputs.CheckboxGroup([...]),  # Same list of symptoms
    ],
    "text",
    description="Select a symptom from the list and click submit to get predicted Disease as the Output. [ NOTE: This app is meant for demo purposes only. Please consult a Doctor if you have any symptoms. ]"
)

iface.launch(share=True)
