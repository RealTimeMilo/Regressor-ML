import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Page configuration
st.set_page_config(page_title="Regression Model Builder", layout="wide")

# Title
st.title("🎯 Regression Model Builder")
st.markdown("Build, train, and evaluate regression models with ease!")

# Sidebar
st.sidebar.header("Configuration")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Overview
    st.header("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with st.expander("View Dataset"):
        st.dataframe(df.head(10))
    
    with st.expander("Dataset Statistics"):
        st.write(df.describe())
    
    # Data Preprocessing
    st.header("🔧 Data Preprocessing")
    
    # Select target variable
    target_col = st.selectbox("Select Target Variable", df.columns)
    
    # Select feature variables
    feature_cols = st.multiselect(
        "Select Feature Variables",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col]
    )
    
    if len(feature_cols) > 0:
        # Handle missing values
        missing_strategy = st.radio(
            "Handle Missing Values",
            ["Drop rows with missing values", "Fill with mean", "Fill with median"]
        )
        
        if missing_strategy == "Drop rows with missing values":
            df = df.dropna()
        elif missing_strategy == "Fill with mean":
            df = df.fillna(df.mean(numeric_only=True))
        else:
            df = df.fillna(df.median(numeric_only=True))
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Train-test split
        st.subheader("Train-Test Split")
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random State", 0, 100, 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Feature scaling
        scale_features = st.checkbox("Scale Features (Standardization)")
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            st.session_state.scaler = scaler
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_cols
        
        st.success(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Model Selection
        st.header("🤖 Model Selection & Training")
        
        model_type = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", 
             "Random Forest", "Support Vector Regression"]
        )
        
        # Model parameters
        col1, col2 = st.columns(2)
        
        with col1:
            if model_type == "Ridge Regression":
                alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0)
            elif model_type == "Lasso Regression":
                alpha = st.slider("Alpha (Regularization)", 0.01, 10.0, 1.0)
            elif model_type == "Random Forest":
                n_estimators = st.slider("Number of Trees", 10, 200, 100)
                max_depth = st.slider("Max Depth", 1, 20, 10)
            elif model_type == "Support Vector Regression":
                C = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                elif model_type == "Lasso Regression":
                    model = Lasso(alpha=alpha)
                elif model_type == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
                else:
                    model = SVR(C=C, kernel=kernel)
                
                model.fit(X_train, y_train)
                st.session_state.model = model
                st.success("✅ Model trained successfully!")
        
        # Model Evaluation
        if st.session_state.model is not None:
            st.header("📈 Model Evaluation")
            
            model = st.session_state.model
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Set Performance")
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                
                st.metric("RMSE", f"{train_rmse:.4f}")
                st.metric("MAE", f"{train_mae:.4f}")
                st.metric("R² Score", f"{train_r2:.4f}")
            
            with col2:
                st.subheader("Test Set Performance")
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                st.metric("RMSE", f"{test_rmse:.4f}")
                st.metric("MAE", f"{test_mae:.4f}")
                st.metric("R² Score", f"{test_r2:.4f}")
            
            # Visualization
            st.subheader("Prediction Plots")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Training set
            ax1.scatter(y_train, y_train_pred, alpha=0.5)
            ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            ax1.set_xlabel("Actual Values")
            ax1.set_ylabel("Predicted Values")
            ax1.set_title("Training Set: Actual vs Predicted")
            ax1.grid(True, alpha=0.3)
            
            # Test set
            ax2.scatter(y_test, y_test_pred, alpha=0.5, color='green')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel("Actual Values")
            ax2.set_ylabel("Predicted Values")
            ax2.set_title("Test Set: Actual vs Predicted")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Residual plot
            st.subheader("Residual Plot")
            residuals = y_test - y_test_pred
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_test_pred, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)
            
            # Make predictions
            st.header("🔮 Make Predictions")
            st.write("Enter feature values to make predictions:")
            
            input_data = {}
            cols = st.columns(3)
            for i, feature in enumerate(feature_cols):
                with cols[i % 3]:
                    input_data[feature] = st.number_input(f"{feature}", value=float(X[feature].mean()))
            
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                if scale_features and st.session_state.scaler is not None:
                    input_scaled = st.session_state.scaler.transform(input_df)
                    prediction = model.predict(input_scaled)[0]
                else:
                    prediction = model.predict(input_df)[0]
                
                st.success(f"### Predicted Value: {prediction:.4f}")
            

else:
    st.info("👈 Please upload a CSV file to get started!")
    st.markdown("""
    ### How to use this app:
    1. Upload your CSV dataset using the sidebar
    2. Select your target variable and features
    3. Configure preprocessing options
    4. Choose a regression model and set parameters
    5. Train the model and evaluate performance
    6. Make predictions on new data
    7. Download your trained model
    """)