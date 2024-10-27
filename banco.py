import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Loan Risk Prediction Analysis", layout="wide")

# Title
st.title("Loan Risk Prediction Analysis")

@st.cache_data
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    return None

def prepare_data_for_feature_selection(data):
    df = data.copy()
    le = LabelEncoder()
    if 'purpose' in df.columns:
        df['purpose'] = le.fit_transform(df['purpose'])
    return df

def get_top_features(X, y, k=4):  # Changed to 4 features
    X_prepared = prepare_data_for_feature_selection(X)
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_prepared, y)
    feature_names = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    return list(feature_names), feature_scores

def preprocess_data(data, selected_features=None, is_training=True):
    df = data.copy()
    le = LabelEncoder()
    if 'purpose' in df.columns:
        df['purpose'] = le.fit_transform(df['purpose'])
    
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if is_training and 'not.fully.paid' in numeric_columns:
        numeric_columns = numeric_columns.drop('not.fully.paid')
    
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    if selected_features is not None:
        if is_training:
            return df[selected_features + ['not.fully.paid']]
        else:
            return df[selected_features]
    return df

def train_model(df, selected_features):
    X = df[selected_features]
    y = df['not.fully.paid']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def display_binary_predictor(data, model, selected_features):
    st.subheader("Quick Yes/No Risk Predictor")
    st.write("Enter client information for the top 4 risk factors:")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    with st.form("binary_predictor_form"):
        for i, feature in enumerate(selected_features):
            with col1 if i % 2 == 0 else col2:
                if feature == 'fico':
                    input_data[feature] = st.slider(
                        "FICO Score",
                        min_value=300,
                        max_value=850,
                        value=700,
                        help="Client's FICO credit score"
                    )
                elif feature == 'int.rate':
                    input_data[feature] = st.slider(
                        "Interest Rate (%)",
                        min_value=0.0,
                        max_value=30.0,
                        value=10.0,
                        step=0.1,
                        help="Proposed loan interest rate"
                    )
                elif feature == 'purpose':
                    input_data[feature] = st.selectbox(
                        "Loan Purpose",
                        options=sorted(data[feature].unique()),
                        help="Purpose of the loan"
                    )
                elif feature == 'log.annual.inc':
                    annual_income = st.number_input(
                        "Annual Income ($)",
                        min_value=10000,
                        max_value=1000000,
                        value=50000,
                        step=1000,
                        help="Client's annual income"
                    )
                    input_data[feature] = np.log(annual_income)
                else:
                    input_data[feature] = st.number_input(
                        feature,
                        value=float(data[feature].mean()),
                        step=0.1
                    )
        
        submitted = st.form_submit_button("Predict Risk (Yes/No)")
    
    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df_processed = preprocess_data(input_df, selected_features, is_training=False)
        prediction = model.predict(input_df_processed)
        prediction_proba = model.predict_proba(input_df_processed)
        
        st.markdown("### Risk Assessment Result")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction[0] == 0:
                st.success("✅ NO RISK - Loan Recommended")
            else:
                st.error("⚠️ RISK DETECTED - Loan Not Recommended")
            
            st.write("#### Confidence Level")
            confidence = max(prediction_proba[0]) * 100
            st.progress(confidence/100, text=f"Confidence: {confidence:.1f}%")

def main():
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    data = process_uploaded_file(uploaded_file)
    
    if data is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(data.head())
        
        X = data.drop('not.fully.paid', axis=1)
        y = data['not.fully.paid']
        selected_features, feature_scores = get_top_features(X, y)
        
        df = preprocess_data(data, selected_features, is_training=True)
        model, X_train, X_test, y_train, y_test, y_pred = train_model(df, selected_features)
        
        st.subheader("Top 4 Most Important Features")
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance Score': feature_scores
        }).sort_values('Importance Score', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=feature_importance, x='Importance Score', y='Feature')
        plt.title('Feature Importance Analysis')
        st.pyplot(fig)
        
        st.markdown("---")
        display_binary_predictor(data, model, selected_features)  # Added new binary predictor
        
        st.markdown("---")
        st.subheader("Model Performance Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        
        with col2:
            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)

if __name__ == "__main__":
    main()