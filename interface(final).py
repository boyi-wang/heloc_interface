#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:06:14 2024

@author: boyi220jessica
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

st.title("HW5")
st.write("Team 15: Rongjiali Ding, Boyi Wang, Mengyao Wang, Siqi Cai, Tianqi Ge")

# Feature weights data
feature_weights = {
    "ExternalRiskEstimate": -0.07119427298761195,
    "MSinceOldestTradeOpen": -0.0009399601555085426,
    "AverageMInFile": -0.006240880552121386,
    "NumSatisfactoryTrades": -0.027014630580453728,
    "PercentTradesNeverDelg": -0.01266954375210325,
    "MaxDelg2PublicRecLast12M": -0.06124044247199626,
    "MaxDelgEver": 0.0011087867583707328,
    "NetFractionRevolvingBurden": 0.008927101131226836,
    "NetFractionInstallBurden": 0.00362501200198794,
    "NumTradesOpeninLast12M": 0.03106552468996071,
    "NumBank2NatlTradesWHighUtilization": 0.10339158486186528,
    "MSinceOldestTradeOpen=-8": 0.14587477201440444,
    "NetFractionInstallBurden=-8": 0.0037582317698964035,
    "NetFractionRevolvingBurden=-8": -0.11302461698957002,
    "NumBank2NatlTradesWHighUtilization=-8": 0.508656748927123
}

# Convert dictionary to DataFrame for better handling in Streamlit
feature_df = pd.DataFrame(list(feature_weights.items()), columns=['Feature', 'Weight'])

# Plotting using Matplotlib
fig, ax = plt.subplots()
ax.barh(feature_df['Feature'], feature_df['Weight'], color='skyblue')
ax.set_xlabel('Feature Importances')
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot in Streamlit
# Use markdown with HTML and style attributes to customize the title size
st.markdown("""
    <h1 style='text-align: center;font-size: 24px;'>Feature Importances for Credit Scoring Model</h1>
    """, unsafe_allow_html=True)
st.pyplot(fig)

# Define the feature information including thresholds and input ranges
feature_info = {
"ExternalRiskEstimate": {"threshold": 72, "min": 0, "max": 100, "message": "A low external risk estimate could indicate that external sources perceive the individual as having a higher credit risk."},
    "MSinceOldestTradeOpen": {"threshold": 200, "min": 0, "max": 1000, "message": "A shorter time since the oldest trade line was opened might suggest a relatively limited credit history, which could be associated with higher risk, especially if other factors are also unfavorable."},
    "AverageMInFile": {"threshold": 78, "min": 0, "max": 400, "message": "A lower average minimum balance might indicate that the user has less financial stability or fewer financial reserves. This can suggest a higher risk of default."},
    "NumSatisfactoryTrades": {"threshold": 21, "min": 0, "max": 100, "message": "A low number of satisfactory trades might indicate a lack of established positive credit history, potentially signaling higher risk."},
    "PercentTradesNeverDelq": {"threshold": 92, "min": 0, "max": 100, "message": "A lower percentage of trades with no delinquency history may suggest a higher likelihood of payment issues or financial instability."},
    "MaxDelq2PublicRecLast12M": {"threshold": 5, "min": 0, "max": 10, "message": "Recent public records indicating delinquency or financial distress within the last 12 months could significantly increase perceived credit risk."},
    "MaxDelqEver": {"threshold": 6, "min": 0, "max": 10, "message": "A history of serious delinquencies or defaults could indicate a higher likelihood of future credit problems."},
    "NetFractionRevolvingBurden": {"threshold": 35, "min": 0, "max": 300, "message": "High revolving credit utilization, where a large portion of available credit is being used, may suggest financial strain and higher risk."},
    "NetFractionInstallBurden": {"threshold": 68, "min": 0, "max": 100, "message": "A high installment loan burden might indicate a heavy debt load, potentially increasing the risk of default."},
    "NumTradesOpeninLast12M": {"threshold": 2, "min": 0, "max": 10, "message": "A high number of recent credit inquiries or newly opened accounts could indicate financial instability or a higher likelihood of future payment difficulties."},
    "NumBank2NatlTradesWHighUtilization": {"threshold": 1, "min": 0, "max": 20, "message": "A high number of trades with high utilization may suggest over-leveraging, which could increase the risk of default."},
    "MSince0ldestTrade0pen=-8": {"threshold": 0.024848, "min": 0, "max": 1, "message":""},
    "NetFractionInstallBurden=-8": {"threshold": 0.343813, "min": 0, "max": 1,"message":""},
    "NetFractionRevolvingBurden=-8": {"threshold": 0.016227, "min": 0, "max": 1,"message":""},
    "NumBank2NatlTradesWHighUtilization=-8": {"threshold": 0.054429, "min": 0, "max": 1,"message":""}
}

def plot_feature_comparison(feature, user_input, threshold):
    fig, ax = plt.subplots()
    categories = [f"Input", f"Threshold"]
    values = [user_input, threshold]
    # Set colors to lighter shades
    colors = ['#add8e6', '#90ee90'] if user_input >= threshold else ['#ffcccb', '#90ee90']
    ax.bar(categories, values, color=colors)
    ax.set_ylabel('Values')
    ax.set_title(f'Comparison for {feature}')
    ax.set_ylim([0, feature_info[feature]['max']])
    return fig

st.markdown("<h1 style='text-align: center; font-size: 24px;'>Feature Analysis and Prediction</h1>", unsafe_allow_html=True)


# Load the model
with open("logistic_regression_model.pkl", 'rb') as f:
    loaded_model = pickle.load(f)




# Display and interact with features
for feature, info in feature_info.items():
    with st.expander(f"Input for {feature}"):
        col1, col2 = st.columns([2, 3])
        with col1:
            # Format the label to include the value range
            label = f"Enter value ({info['min']}-{info['max']})"
            user_input = st.number_input(label, min_value=info["min"], max_value=info["max"], value=int(info["threshold"] * 0.9), key=f"input_{feature}")
            message = info.get('message', 'No specific message provided for this feature.')  # Default message if 'message' key is missing
            if user_input < info["threshold"]:
                st.error(message)  # Use the message from the dictionary or the default
        with col2:
            fig = plot_feature_comparison(feature, user_input, info["threshold"])
            st.pyplot(fig)
    


# Prediction button
if st.button("Predict"):
    # Retrieve inputs using the keys used in number_input widgets
    inputs = [st.session_state[f"input_{feature}"] for feature, _ in feature_info.items()]
    # Get the probabilities of each class
    probabilities = loaded_model.predict_proba([inputs])[0]
    # Get the class prediction directly
    prediction = loaded_model.predict([inputs])[0]

    if prediction == 1:
        # Use HTML and CSS to style the output for a prediction of 0
        st.markdown(f"""
            <div style='background-color: #ffcccb; padding: 20px; border-radius: 10px;'>
                <h2 style='color: #990000; text-align: center;'>YOU'RE INELIGIBLE</h2>
                <p style='color: #660000; font-size: 16px; text-align: center;'>
                    Based on your responses, we believe that you are not currently eligible for the application. You are {probabilities[1] * 100:.2f}% likely to be ineligible.
                    Please check back with us if you perform any product or software development-related work in the future.
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif prediction == 0:
        # Use HTML and CSS to style the output for a prediction of 1
        st.markdown(f"""
            <div style='background-color: #90ee90; padding: 20px; border-radius: 10px;'>
                <h2 style='color: #006400; text-align: center;'>CONGRATULATIONS!</h2>
                <p style='color: #004d00; font-size: 16px; text-align: center;'>
                    Congratulations! Based on your responses, you are {probabilities[0] * 100:.2f}% likely to be eligible for the application.
                    We look forward to your participation.
                </p>
            </div>
        """, unsafe_allow_html=True)
