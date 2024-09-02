# # app.py

# import gradio as gr
# import pandas as pd
# import numpy as np
# from joblib import load

# def predict_disease_from_symptom(symptom_list):
#     symptoms = { ... }  # Same dictionary from previous example

#     for s in symptom_list:
#         symptoms[s] = 1

#     df_test = pd.DataFrame(columns=list(symptoms.keys()))
#     df_test.loc[0] = np.array(list(symptoms.values()))

#     clf = load(str("./saved_model/random_forest.joblib"))
#     result = clf.predict(df_test)

#     del df_test

#     return f"{result[0]}"

# iface = gr.Interface(
#     predict_disease_from_symptom,
#     [
#         gr.inputs.CheckboxGroup([...]),  # Same list of symptoms
#     ],
#     "text",
#     description="Select a symptom from the list and click submit to get predicted Disease as the Output. [ NOTE: This app is meant for demo purposes only. Please consult a Doctor if you have any symptoms. ]"
# )

# iface.launch(share=True)
import gradio as gr
import pandas as pd
import numpy as np
from joblib import load

def predict_disease_from_symptom(symptom_list):
    # Define your symptoms dictionary here (replace ... with your actual symptom mapping)
    symptoms = { ... }

    # Update symptoms based on selected checkboxes
    for s in symptom_list:
        if s in symptoms:
            symptoms[s] = 1  # Assuming 1 means the symptom is present

    # Create a DataFrame with columns from symptoms dictionary
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))

    # Load your trained model
    try:
        clf = load("./saved_model/random_forest.joblib")
    except Exception as e:
        return f"Error loading model: {str(e)}"

    # Make predictions using the loaded model
    result = clf.predict(df_test)

    # Clean up
    del df_test

    # Return the predicted disease
    return f"{result[0]}"

# Define Gradio interface
iface = gr.Interface(
    predict_disease_from_symptom,
    [
        gr.inputs.CheckboxGroup([...]),  # List of symptoms (checkboxes)
    ],
    "text",  # Output format
    description="Select symptoms and click submit to predict the disease. [ NOTE: This app is for demo purposes only. Consult a doctor for medical advice. ]"
)

# Launch the Gradio interface
iface.launch(share=True)
