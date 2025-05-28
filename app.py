
#Maybe get rid of Customer Profile Being Analyzed in Raw Form if we can get that date in the sidebar always

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
from sklearn.inspection import PartialDependenceDisplay


import streamlit as st
import pandas as pd
import scipy.special

from PIL import Image


# Set full app background to black and text to white
st.markdown(
    """
    <style>
        /* Set background color for entire app */
        .stApp {
            background-color: black;
            color: white;
        }

        /* Optional: Set sidebar background if using sidebar */
        .css-1d391kg {
            background-color: #111111 !important;
        }

        /* Optional: Override widget label colors */
        label, .css-10trblm, .css-1cpxqw2 {
            color: white !important;
        }

        /* Optional: Set input box background and text color */
        .stTextInput > div > div > input {
            background-color: #222;
            color: white;
        }

        .stSelectbox > div > div > div > div {
            background-color: #222;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# Load images once, preferably at the top of app
no_churn_img = Image.open("no_churn.png")
churn_img = Image.open("churn.png")
pande_system_img = Image.open("predictandexplainsystem.png")
telco_header_img = Image.open("telcoheader.png")
telco_header_img = telco_header_img.resize((1000, 400))  # width, height in pixels
customerdata_header_img = Image.open("customerdatainput.png")

# Sample input data for dropdowns (optional)
try:
    X_sample = pd.read_csv("X_sample.csv")  # Or use any small sample of original training data
except:
    X_sample = pd.DataFrame()  # fallback


#Functions
##########################################################################


# Step 1: Define a mapping of feature-value pairs to user-friendly labels
def format_label(feature, value):
    label_map = {
        "gender": {
            1: "Gender: Male",
            0: "Gender: Female"
        },
        "Contract_Month-to-month": {
            1: "Contract: Month-to-month",
            0: "Contract: Not Month-to-month"
        },
        "PaymentMethod_Electronic_check": {
            1: "Payment Method: Electronic Check",
            0: "Payment Method: Not Electronic Check"
        },
        "PaymentMethod_Mailed_check": {
            1: "Payment Method: Mailed Check",
            0: "Payment Method: Not Mailed Check"
        },
        "PaymentMethod_Bank_transfer_(automatic))": {
            1: "Payment Method: Automatic Bank Transfer",
            0: "Payment Method: Not Automatic Bank Transfer"
        },
        "PaymentMethod_Credit_card_(automatic))": {
            1: "Payment Method: Automatic Credit Card",
            0: "Payment Method: Not Automatic Credit Card"
        },
        "Contract_One_year": {
            1: "Contract: One Year",
            0: "Contract: Not One Year"
        },
        "Contract_Two_year": {
            1: "Contract: Two Years",
            0: "Contract: Not Two Years"
        },
        "SeniorCitizen": {
            1: "Senior Citizen: Yes",
            0: "Senior Citizen: No"
        },
        "Partner": {
            1: "have a partner: Yes",
            0: "have a partner: No"
        },
        "Dependents": {
            1: "have dependents: Yes",
            0: "have dependents: No"
        },
        "PhoneService": {
            1: "Phone Service: Yes",
            0: "Phone Service: No"
        },
        "PaperlessBilling": {
            1: "Paperless Billing: Yes",
            0: "Paperless Billing: No"
        },
        "InternetService_DSL": {
            1: "DSL Internet Service: Yes",
            0: "DSL Internet Service: No"
        },
        "InternetService_Fiber_optic": {
            1: "Fiber Optic Internet Service: Yes",
            0: "Fiber Optic Internet Service: No"
        },
        "InternetService_No": {
            0: "Internet Service: Yes",
            1: "Internet Service: No"
        },
        "MultipleLines_Yes": {
            1: "Multiple Phone Lines: Yes",
            0: "Multiple Phone Lines: No"
        },
        "MultipleLines_No": {
            0: "Multiple Phone Lines: Yes",
            1: "Multiple Phone Lines: No"
        },
        "OnlineSecurity_Yes": {
            1: "Online Security: Yes",
            0: "Online Security: No"
        },
        "OnlineSecurity_No": {
            0: "Online Security: Yes",
            1: "Online Security: No"
        },
        "OnlineBackup_Yes": {
            1: "Online Backup: Yes",
            0: "Online Backup: No"
        },
        "OnlineBackup_No": {
            0: "Online Backup: Yes",
            1: "Online Backup: No"
        },
        "DeviceProtection_Yes": {
            1: "Device Protection: Yes",
            0: "Device Protection: No"
        },
        "DeviceProtection_No": {
            0: "Device Protection: Yes",
            1: "Device Protection: No"
        },
        "TechSupport_Yes": {
            1: "Tech Support: Yes",
            0: "Tech Support: No"
        },
        "TechSupport_No": {
            0: "Tech Support: Yes",
            1: "Tech Support: No"
        },
        "StreamingTV_Yes": {
            1: "Streaming TV: Yes",
            0: "Streaming TV: No"
        },
        "StreamingTV_No": {
            0: "Streaming TV: Yes",
            1: "Streaming TV: No"
        },
        "StreamingMovies_Yes": {
            1: "Streaming Movies: Yes",
            0: "Streaming Movies: No"
        },
        "StreamingMovies_No": {
            0: "Streaming Movies: Yes",
            1: "Streaming Movies: No"
        },

        # Add other features as needed
    }

    # Return mapped label if defined, otherwise fallback
    if feature in label_map:
        return label_map[feature].get(value, f"{feature}: {value}")
    else:
        return f"{feature}: {value}"




def plot_partial_dependence(model, X, feature, feature_name=None):
    """Plots a PDP for a given feature and displays it in Streamlit."""
    fig, ax = plt.subplots(figsize=(8, 5))
    PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
    title = f"Partial Dependence Plot: {feature_name or feature}"
    st.pyplot(fig)


def get_user_input():
    st.sidebar.header("Manual Customer Input Features")

    # Binary / numeric inputs
    gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
    senior = st.sidebar.selectbox("Senior Citizen (0 = Not, 1 = Senior Citizen)", [0, 1])  # Already numeric
    partner = st.sidebar.selectbox("Has Partner", ['No', 'Yes'])
    dependents = st.sidebar.selectbox("Has Dependents", ['No', 'Yes'])
    phone_service = st.sidebar.selectbox("Has Phone Service", ['No', 'Yes'])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ['No', 'Yes'])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0, step=1.0)

    if tenure == 0:
        total_charges = 0.0
    else:
        total_charges = st.sidebar.slider("Total Charges", 0.0, 15000.0, 1000.0, step=10.0)

    # One-hot encoded categorical options
    online_security = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.sidebar.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    device_protection = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.sidebar.selectbox("Payment Method", [
        'Bank transfer (automatic)', 'Credit card (automatic)',
        'Electronic check', 'Mailed check'])

    # Encode binary fields
    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    input_dict = {
        'gender': binary_map[gender],
        'SeniorCitizen': senior,
        'Partner': binary_map[partner],
        'Dependents': binary_map[dependents],
        'PhoneService': binary_map[phone_service],
        'PaperlessBilling': binary_map[paperless_billing],
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    # One-hot features to generate
    one_hot_fields = {
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'Contract': contract,
        'PaymentMethod': payment_method,
    }

    # Generate one-hot encoded columns
    for feature, value in one_hot_fields.items():
        col_name = f"{feature}_{value.replace(' ', '_')}"
        input_dict[col_name] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Add any missing columns with 0
    model_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'OnlineSecurity_No', 'OnlineSecurity_No_internet_service',
        'OnlineSecurity_Yes', 'OnlineBackup_No',
        'OnlineBackup_No_internet_service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No_internet_service',
        'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_No_internet_service', 'TechSupport_Yes', 'StreamingTV_No',
        'StreamingTV_No_internet_service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No_internet_service',
        'StreamingMovies_Yes', 'MultipleLines_No',
        'MultipleLines_No_phone_service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber_optic',
        'InternetService_No', 'Contract_Month-to-month', 'Contract_One_year',
        'Contract_Two_year', 'PaymentMethod_Bank_transfer_(automatic)',
        'PaymentMethod_Credit_card_(automatic)',
        'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check'
    ]

    # Ensure all model columns are present in the input
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match model
    input_df = input_df[model_columns]

    return input_df


# Add this new function to render inputs from a sample row (Series or dict)
def get_user_input_from_sample(sample_row):
    st.sidebar.header("Sample Customer Input Features")

    # Use .get with fallback for safety in case of missing columns

    gender_val = 'Male' if sample_row.get('gender', 0) == 1 else 'Female'
    gender = st.sidebar.selectbox("Gender", ['Female', 'Male'], index=0 if gender_val == 'Female' else 1)

    senior = st.sidebar.selectbox("Senior Citizen (0 = Not, 1 = Senior Citizen)", [0, 1], index=sample_row.get('SeniorCitizen', 0))

    partner_val = 'Yes' if sample_row.get('Partner', 0) == 1 else 'No'
    partner = st.sidebar.selectbox("Has Partner", ['No', 'Yes'], index=0 if partner_val == 'No' else 1)

    dependents_val = 'Yes' if sample_row.get('Dependents', 0) == 1 else 'No'
    dependents = st.sidebar.selectbox("Has Dependents", ['No', 'Yes'], index=0 if dependents_val == 'No' else 1)

    phone_service_val = 'Yes' if sample_row.get('PhoneService', 0) == 1 else 'No'
    phone_service = st.sidebar.selectbox("Has Phone Service", ['No', 'Yes'], index=0 if phone_service_val == 'No' else 1)

    paperless_billing_val = 'Yes' if sample_row.get('PaperlessBilling', 0) == 1 else 'No'
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ['No', 'Yes'], index=0 if paperless_billing_val == 'No' else 1)

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, int(sample_row.get('tenure', 12)))

    monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 200.0, float(sample_row.get('MonthlyCharges', 70.0)), step=1.0)

    total_charges_default = float(sample_row.get('TotalCharges', 0.0))
    if tenure == 0:
        total_charges = 0.0
    else:
        total_charges = st.sidebar.slider("Total Charges", 0.0, 15000.0, total_charges_default, step=10.0)

    def get_cat_value(sample_row, feature_prefix, options):
        # Find which option in `options` matches a column with 1
        for opt in options:
            col_name = f"{feature_prefix}_{opt.replace(' ', '_')}"
            if sample_row.get(col_name, 0) == 1:
                return opt
        # fallback
        return options[0]

    online_security = st.sidebar.selectbox("Online Security", ['No', 'Yes', 'No internet service'],
                                           index=['No', 'Yes', 'No internet service'].index(
                                               get_cat_value(sample_row, 'OnlineSecurity', ['No', 'Yes', 'No internet service'])
                                           ))
    online_backup = st.sidebar.selectbox("Online Backup", ['No', 'Yes', 'No internet service'],
                                        index=['No', 'Yes', 'No internet service'].index(
                                            get_cat_value(sample_row, 'OnlineBackup', ['No', 'Yes', 'No internet service'])
                                        ))
    device_protection = st.sidebar.selectbox("Device Protection", ['No', 'Yes', 'No internet service'],
                                            index=['No', 'Yes', 'No internet service'].index(
                                                get_cat_value(sample_row, 'DeviceProtection', ['No', 'Yes', 'No internet service'])
                                            ))
    tech_support = st.sidebar.selectbox("Tech Support", ['No', 'Yes', 'No internet service'],
                                        index=['No', 'Yes', 'No internet service'].index(
                                            get_cat_value(sample_row, 'TechSupport', ['No', 'Yes', 'No internet service'])
                                        ))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'],
                                       index=['No', 'Yes', 'No internet service'].index(
                                           get_cat_value(sample_row, 'StreamingTV', ['No', 'Yes', 'No internet service'])
                                       ))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'],
                                            index=['No', 'Yes', 'No internet service'].index(
                                                get_cat_value(sample_row, 'StreamingMovies', ['No', 'Yes', 'No internet service'])
                                            ))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'],
                                          index=['No', 'Yes', 'No phone service'].index(
                                              get_cat_value(sample_row, 'MultipleLines', ['No', 'Yes', 'No phone service'])
                                          ))
    internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'],
                                            index=['DSL', 'Fiber optic', 'No'].index(
                                                get_cat_value(sample_row, 'InternetService', ['DSL', 'Fiber optic', 'No'])
                                            ))
    contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'],
                                    index=['Month-to-month', 'One year', 'Two year'].index(
                                        get_cat_value(sample_row, 'Contract', ['Month-to-month', 'One year', 'Two year'])
                                    ))
    payment_method = st.sidebar.selectbox("Payment Method", [
        'Bank transfer (automatic)', 'Credit card (automatic)',
        'Electronic check', 'Mailed check'],
        index=[
            'Bank transfer (automatic)', 'Credit card (automatic)',
            'Electronic check', 'Mailed check'
        ].index(
            get_cat_value(sample_row, 'PaymentMethod', [
                'Bank transfer (automatic)', 'Credit card (automatic)',
                'Electronic check', 'Mailed check'
            ])
        ))

    # Encode binary fields
    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    input_dict = {
        'gender': binary_map[gender],
        'SeniorCitizen': senior,
        'Partner': binary_map[partner],
        'Dependents': binary_map[dependents],
        'PhoneService': binary_map[phone_service],
        'PaperlessBilling': binary_map[paperless_billing],
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    one_hot_fields = {
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'Contract': contract,
        'PaymentMethod': payment_method,
    }

    for feature, value in one_hot_fields.items():
        col_name = f"{feature}_{value.replace(' ', '_')}"
        input_dict[col_name] = 1

    # Add missing model columns with 0
    model_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'OnlineSecurity_No', 'OnlineSecurity_No_internet_service',
        'OnlineSecurity_Yes', 'OnlineBackup_No',
        'OnlineBackup_No_internet_service', 'OnlineBackup_Yes',
        'DeviceProtection_No', 'DeviceProtection_No_internet_service',
        'DeviceProtection_Yes', 'TechSupport_No',
        'TechSupport_No_internet_service', 'TechSupport_Yes', 'StreamingTV_No',
        'StreamingTV_No_internet_service', 'StreamingTV_Yes',
        'StreamingMovies_No', 'StreamingMovies_No_internet_service',
        'StreamingMovies_Yes', 'MultipleLines_No',
        'MultipleLines_No_phone_service', 'MultipleLines_Yes',
        'InternetService_DSL', 'InternetService_Fiber_optic',
        'InternetService_No', 'Contract_Month-to-month', 'Contract_One_year',
        'Contract_Two_year', 'PaymentMethod_Bank_transfer_(automatic)',
        'PaymentMethod_Credit_card_(automatic)',
        'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check'
    ]

    for col in model_columns:
        if col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]  # reorder columns

    return input_df



#Start App
###################################################

# Load the trained model
model = joblib.load("telco_churn_xgb_model.pkl")
X = pd.read_csv("X_processed.csv")

#Header Image
st.image(pande_system_img, use_container_width=True)

#Sub Header Image
st.image(telco_header_img, use_container_width=True)


# st.title("Customer Churn Predictor & Explainer")

#sidebar Header Image
st.sidebar.image(customerdata_header_img, use_container_width=True)

# Sidebar input: select existing row or manual input
input_method = st.sidebar.radio("Select Your Input Method Here", ["Manual Input", "Select Sample Row"])

# --- Feature Input ---
if input_method == "Manual Input":
    input_df = get_user_input()



elif input_method == "Select Sample Row":
    if X_sample.empty:
        st.warning("No sample data available.")
        st.stop()
    else:
        row_index = st.sidebar.number_input("Select Row", 0, len(X_sample) - 1, 0)
        selected_row = X_sample.iloc[row_index]

        st.sidebar.markdown("### Sample Row Features:")
        for feature, value in selected_row.items():
            readable_label = format_label(feature, value)
            st.sidebar.markdown(f"- {readable_label}")

        # Keep it in DataFrame format for model input
        input_df = pd.DataFrame([selected_row])


else:
    st.warning("No sample data available.")
    st.stop()

# st.subheader("Customer Profile Being Analyzed (In Raw Form):")
# st.write(input_df)

# --- Prediction ---
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result:")
# Display result text
if prediction == 0:
    st.image(no_churn_img, caption="Customer is likely to stay!", use_container_width=True)
else:
    st.image(churn_img, caption="Customer might leave :(", use_container_width=True)

st.write(f"Prediction: **{'Churn' if prediction == 1 else 'No Churn'}**")
st.write(f"Probability of Churn: **{100*proba:.2f}%**")




# --- Explanation using SHAP ----------------------------
####################################################
#st.subheader("Explanation")

# Compute SHAP values
explainer = shap.TreeExplainer(model, X, feature_perturbation='interventional')#conditional distribution
#explainer = shap.TreeExplainer(model)#marginal distribution
shap_values = explainer.shap_values(input_df)

#Extract Top Features Contributing to the Prediction

# Get feature names and their SHAP values for this input
feature_names = input_df.columns
shap_vals = shap_values[0]  # assumes only 1 input row


# Create a DataFrame of features and their SHAP contributions
shap_df = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_vals,
    "value": input_df.iloc[0].values
})




# Mapping technical feature names to human-readable phrases
feature_name_map = {
    'tenure': 'have been a customer for this many months:',
    'MonthlyCharges': 'have monthly charges of $',
    'TotalCharges': 'have total charges of $',
    'gender_Female': ('are female', 'are not female'),
    'gender_Male': ('are male','are not male'),
    'SeniorCitizen': ("are a senior citizen", "are not a senior citizen"),
    'Partner': ("have a partner", "do not have a partner"),
    'Dependents': ("have dependents", "do not have dependents"),
    'PhoneService': ("have phone service", "do not have phone service"),
    'MultipleLines_No_phone_service': 'has no phone service',
    'MultipleLines_Yes': ("have multiple phone lines", "do not have multiple phone lines"),
    'InternetService_DSL': ("have DSL internet", "do not have DSL internet"),
    "InternetService_Fiber_optic": ("have fiber optic internet", "do not have fiber optic internet"),
    'InternetService_No': ('do not have internet service','have internet service'),
    'OnlineSecurity_Yes': ('have online security','do not have online security'),
    'OnlineSecurity_No': ('do not have online security','have online security'),    
    'OnlineSecurity_No_internet_service': 'do not have internet service to have online security',

    'OnlineBackup_Yes': ('have online backup','do not have online backup'),
    'OnlineBackup_No': ('do not have online backup','have online backup'),
    'OnlineBackup_No_internet_service': 'do not have internet service to have online backup',

    'DeviceProtection_Yes': ('have device protection','do not have device protection'),
    'DeviceProtection_No': ('do not have device protection','have device protection'),
    'DeviceProtection_No_internet_service': 'do not have internet service to have device protection',

    'TechSupport_Yes': ('have tech support', 'do not have tech support'),
    'TechSupport_No': ('do not have tech support', 'have tech support'),
    'TechSupport_No_internet_service': 'do not have internet service to have tech support',

    'StreamingTV_Yes': ('stream TV', 'do not stream TV'),
    'StreamingTV_No': ('do not stream TV', 'stream TV'),
    'StreamingTV__No_internet_service': 'do not have internet service to have streaming tv',

    'StreamingMovies_Yes': ('stream movies', 'do not stream movies'),
    'StreamingMovies_No': ('do not stream movies', 'stream movies'),
    'StreamingMovies__No_internet_service': 'do not have internet service to have streaming movies',

    'Contract_Month-to-month': ("have a month-to-month contract", "do not have a month-to-month contract"),
    'Contract_One_year': ('have a one-year contract', "do not have a one-year contract"),
    'Contract_Two_year': ('have a two-year contract', "do not have a two-year contract"),
    "PaperlessBilling": ("use paperless billing", "do not use paperless billing"),
    'PaymentMethod_Electronic_check': ('use electronic check as a payment method','do not use electronic check as a payment method'),
    'PaymentMethod_Mailed_check': ('mail a check as a payment method','do not mail a check as a payment method'),
    'PaymentMethod_Bank_transfer_(automatic)': ('use automatic bank transfer','do not use automatic bank transfer'),
    'PaymentMethod_Credit_card_(automatic)': ('use automatic credit card payment','do not use automatic credit card payment')
}

# Generate plain-English explanation
shap_vals = shap_values[0]
expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[0] # baseline log-odds or probability
top_n = 5
sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]

log_odds_pred = expected_value + shap_vals.sum()
prob_pred = scipy.special.expit(log_odds_pred)

explanations = []
for idx in sorted_indices:
    feature_name = input_df.columns[idx]
    shap_value = shap_vals[idx]
    feature_val = input_df.iloc[0, idx]

    # Convert SHAP value to estimated change in probability
    prob_change = shap_value
    impact_percent = round(abs(prob_change) * 100, 1)
    direction = "❗ MORE 🔺 likely to churn" if shap_value > 0 else " ✅ LESS ⬇️ likely to churn"

    # Try to find human-readable version
    human_name = feature_name_map.get(feature_name)
    explanation = ""  # Initialize default


    # 🟡 Handle these important numeric features first
    if feature_name in ["tenure", "MonthlyCharges", "TotalCharges"]:
        value = round(float(feature_val), 2)

        if feature_name == "tenure":
            explanation = f"The customer is {impact_percent}% {direction} because they have been a customer for {int(value)} month(s)."
        elif feature_name == "MonthlyCharges":
            explanation = f"The customer is {impact_percent}% {direction} because they pay ${value} per month."
        elif feature_name == "TotalCharges":
            explanation = f"The customer is {impact_percent}% {direction} because they have total charges of ${value}."
    else:

    # Try to find human-readable version or tuple
        human = feature_name_map.get(feature_name)

        if human:
            if isinstance(human, tuple):
                # Handle binary features using tuple of (positive_text, negative_text)
                explanation = f"The customer is {impact_percent}% {direction} because they {human[0] if feature_val == 1 else human[1]}."
            else:
                # For normal mapped features
                explanation = f"The customer is {impact_percent}% {direction} because they {human}."

        else:
            # Fallback for unmapped features
            clean_name = feature_name.replace('_', ' ').lower()

            if isinstance(feature_val, (int, float)) and feature_val in [0, 1]:
                verb = "have" if feature_val == 1 else "do not have"
                explanation = f"The customer is {impact_percent}% {direction} because they {verb} {clean_name}."
            else:
                explanation = f"The customer is {impact_percent}% {direction} because of {clean_name} ({feature_val})."

    explanations.append(explanation)

# Display
st.markdown(
    """
    ### 🧠 Prediction Explanations:
    
    This explanation shows which customer features most affect their chance of churning.  
    The features are ranked by how much they increase or decrease the risk for this customer.

    """
)
for exp in explanations:
    st.markdown(f"- {exp}")








#Create Image Plots for dynamic SHAP values_____________________________________
####################################################



import matplotlib.pyplot as plt

st.write("### Local Feature Importances (SHAP Values)")

# Display
st.markdown(
    """
    #### 🔍 What are Local SHAP Values?
SHAP (SHapley Additive exPlanations) values help explain why the model made a specific prediction for this individual user. 
Each feature gets a score showing how much it pushed the prediction up or down compared to the average prediction. 
Positive values increase the chance of churn; negative values decrease it. 
This helps you understand which details about this customer mattered most to the model's decision.

    """
)

#Interactive Plotly plot
####################################################
import plotly.express as px




# Sort by absolute SHAP value (most impactful features first)
shap_df["abs_shap"] = shap_df["shap_value"].abs()
shap_df_sorted = shap_df.sort_values(by="abs_shap", ascending=False).head(10)  # Top 15 features



# Step 2: Apply the mapping
shap_df_sorted["label"] = shap_df_sorted.apply(
    lambda row: format_label(row["feature"], row["value"]),
    axis=1
)


# Plotly bar chart
fig = px.bar(
    shap_df_sorted,
    x="shap_value",
    y="label",
    orientation="h",
    title="Top SHAP Feature Contributions to Prediction",
    labels={"shap_value": "SHAP Value", "label": "Feature"},
    color="shap_value",
    color_continuous_scale="RdYlGn_r",
    range_color=[-max(shap_df_sorted["abs_shap"]), max(shap_df_sorted["abs_shap"])],
    text=shap_df_sorted["shap_value"].map(lambda x: f"{100*x:.1f}%"),
)

#fig.update_traces(textposition="outside")
#fig.update_traces(textposition="inside", insidetextanchor="start", textfont_color="black")
fig.update_traces(
    textposition="inside",
    insidetextanchor="start",
    textfont=dict(
        color="black",         # Text color inside bars
        size=20,               # Font size
        family="Arial Black"   # Bold font (try Helvetica, Verdana, etc. too)
    )
)

fig.update_layout(
    yaxis=dict(autorange="reversed"),  # Highest impact at top
    # plot_bgcolor="white",
    showlegend=False,
    uniformtext_minsize=8,
      #uniformtext_mode='hide',
    plot_bgcolor='black',   # Background inside the chart (plot area)
    paper_bgcolor='black',  # Background outside the plot (full figure)
    font=dict(color='white')  # Optional: Change font color to white for contrast
    
)
fig.update_coloraxes(showscale=False)
st.plotly_chart(fig, use_container_width=True)









# Global feature importance
############################################################
# st.write("### Global Feature Importance")
# st.bar_chart(pd.Series(model.feature_importances_, index=model.get_booster().feature_names))
st.write("### Global Feature Importance (SHAP)")


# Display
st.markdown(
    """
    #### 🌍 What are Global SHAP Values?
Global SHAP values show which features the model relies on most often across all customers.
Instead of focusing on a single prediction, this view helps you understand the overall behavior of the model—which features are generally most influential in predicting churn. 
The longer the bar, the bigger the impact that feature tends to have.

    """
)

feature_importance_series = pd.Series(model.feature_importances_, index=model.get_booster().feature_names)
df_importance = feature_importance_series.sort_values(ascending=False).reset_index()
df_importance.columns = ['feature', 'importance']

fig = px.bar(
    df_importance,
    x='importance',
    y='feature',
    orientation='h',
    # text='importance',
    text=df_importance['importance'].map(lambda x: f"{100*x:.2f}%"),
    title='Top Global Feature Importances',
    labels={'importance': 'Importance', 'feature': 'Feature'}
)

#fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
fig.update_traces(
    # texttemplate='%{text:.1f}',
    textposition="inside",
    insidetextanchor="start",
    textfont=dict(
        color="black",         # Text color inside bars
        size=20,               # Font size
        family="Arial Black"   # Bold font (try Helvetica, Verdana, etc. too)
    )
)
#fig.update_layout(yaxis=dict(autorange="reversed"))
fig.update_layout(
    yaxis=dict(autorange="reversed"),  # Highest impact at top
    # plot_bgcolor="white",
    font=dict(size=16, color = 'white'),  # Increase global font size
    title_font=dict(size=20),
    margin=dict(l=120, r=40, t=60, b=40),  # Add left margin for long feature names
    height=1400,  # Make the whole chart taller
    showlegend=False,
    uniformtext_minsize=8,
      #uniformtext_mode='hide',
    plot_bgcolor='black',   # Background inside the chart (plot area)
    paper_bgcolor='black',  # Background outside the plot (full figure)
    # font=dict(color='white')  # Optional: Change font color to white for contrast
    
)

st.plotly_chart(fig, use_container_width=True)


# Conceptual PDPs
###################################################################



st.subheader("📈 Partial Dependence Plots")


st.markdown("""
Partial Dependence Plots show how a single feature affects the model's prediction on average, while keeping all other features constant. It answers the question:
"If this one feature changes, how does the predicted outcome typically change?"

This helps you understand the general relationship between a feature 
            (like having a month-to-month contract increases the probability of churning while having a yearly contract decreases the probability of churning)
             and the predicted probability of churn, regardless of the specific customer.
""")

# Dropdown for selecting a feature
feature_options = X.columns.tolist()
selected_feature = st.selectbox("Select a feature to explore:", feature_options)

fig, ax = plt.subplots(figsize=(8, 5))

display = PartialDependenceDisplay.from_estimator(
    model,
    X,
    [selected_feature],    
    line_kw={"color": "cyan"},           # Line color
    ax=ax,
)

# Get the actual axes object created by PDP
ax = display.axes_[0, 0]

# Force black background and white text AFTER the plot is created
fig.patch.set_facecolor('black')         # Full figure background
ax.set_facecolor('black')                # Inner plot background
ax.title.set_color('white')              # Title
ax.xaxis.label.set_color('white')        # X-axis label
ax.yaxis.label.set_color('white')        # Y-axis label
ax.tick_params(colors='white')           # Tick marks
for spine in ax.spines.values():         # Plot borders
    spine.set_color('white')

# Optional: Change line and marker color (e.g., red or cyan)
for line in ax.get_lines():
    line.set_color('cyan')

# Show in Streamlit
st.pyplot(fig)


# Show the PDP
#plot_partial_dependence(model, X, selected_feature)
