import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging to suppress LightGBM output
logging.getLogger('lightgbm').setLevel(logging.ERROR)

file_path = 'Hospital_Rating_Dataset.csv'  # Replace with actual file path
  # Replace with actual file path

# Loading and preprocessing the dataset
def load_and_preprocess_data(file_path):
    # Using pandas for simplicity
    df = pd.read_csv(file_path)
    
    # Replacing 'Not Applicable' with NaN
    df = df.replace('Not Applicable', np.nan)
    
    # Converting relevant columns to numeric
    numeric_cols = ['HCAHPS_Answer_Percent', 'HCAHPS_Linear_Mean_Value', 
                    'Number_of_Completed_Surveys', 'Survey_Response_Rate_Percent']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Pivoting data to have one row per hospital
    pivot_df = df.pivot_table(
        index=['facility_id', 'facility_name'],
        columns='HCAHPS_measure_id',
        values=['HCAHPS_Answer_Percent', 'HCAHPS_Linear_Mean_Value'],
        aggfunc='mean'
    )
    
    # Flattening multi-level column index
    pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_df.columns]
    
    # Extracting target variable
    target = df[df['HCAHPS_measure_id'] == 'H_STAR_RATING'][['facility_id', 'Patient_Survey_Star_Rating']]
    target['Patient_Survey_Star_Rating'] = pd.to_numeric(target['Patient_Survey_Star_Rating'], errors='coerce')
    
    # Adjusting target to be 0-indexed for XGBoost compatibility
    target['Patient_Survey_Star_Rating'] = target['Patient_Survey_Star_Rating'] - 1
    
    # Merging pivoted features with target
    final_df = pivot_df.merge(target, on='facility_id')
    
    # Adding survey counts and response rate as features
    survey_cols = df[['facility_id', 'Number_of_Completed_Surveys', 'Survey_Response_Rate_Percent']].groupby('facility_id').mean()
    final_df = final_df.merge(survey_cols, on='facility_id')
    
    return final_df
    
    return final_df

# Comprehensive feature engineering
def engineer_features(df):
    # Verify required columns exist
    required_cols = ['Number_of_Completed_Surveys', 'Survey_Response_Rate_Percent', 'Patient_Survey_Star_Rating']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in the dataset")
            # Create placeholder columns with zeros if not present
            df[col] = 0
    
    # Aggregating "Always" responses
    always_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'A_P' in col]
    df['avg_always_percent'] = df[always_cols].mean(axis=1)
    df['max_always_percent'] = df[always_cols].max(axis=1)
    
    # Aggregating "Sometimes/Never" responses
    sn_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'SN_P' in col]
    df['avg_sn_percent'] = df[sn_cols].mean(axis=1)
    
    # Category-specific features
    nurse_cols = [col for col in df.columns if 'H_NURSE' in col and 'HCAHPS_Answer_Percent' in col]
    doctor_cols = [col for col in df.columns if 'H_DOCTOR' in col and 'HCAHPS_Answer_Percent' in col]
    df['avg_nurse_percent'] = df[nurse_cols].mean(axis=1)
    df['avg_doctor_percent'] = df[doctor_cols].mean(axis=1)
    
    # Interaction features
    df['nurse_doctor_interaction'] = df['avg_nurse_percent'] * df['avg_doctor_percent']
    
    # Weighted features by survey response rate
    df['weighted_rating'] = df['Patient_Survey_Star_Rating'] * df['Survey_Response_Rate_Percent'] / 100
    
    # Normalizing survey volume
    df['normalized_surveys'] = df['Number_of_Completed_Surveys'] / df['Number_of_Completed_Surveys'].max()
    
    # Ratio features
    df['always_to_sn_ratio'] = df['avg_always_percent'] / (df['avg_sn_percent'] + 1e-5)
    
    return df
    
    return df

# Preprocessing and feature selection
def preprocess_and_select_features(X, y):
    # KNN imputation for missing values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Capping outliers
    for col in X_imputed.columns:
        if 'Percent' in col:
            X_imputed[col] = X_imputed[col].clip(lower=0, upper=100)
    
    # Feature selection with SelectKBest
    selector = SelectKBest(f_classif, k=50)  # Select top 50 features
    X_selected = selector.fit_transform(X_imputed, y)
    selected_features = X_imputed.columns[selector.get_support()].tolist()
    
    return X_selected, selected_features

# Training and evaluating the model
def train_and_evaluate(X, y, selected_features):
    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initializing models
    xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
    lgbm_model = LGBMClassifier(random_state=42, verbose=-1)  # Set verbose to -1 to suppress warnings
    rf_model = RandomForestClassifier(random_state=42)
    
    # Hyperparameter tuning for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    xgb_search.fit(X_train_scaled, y_train)
    best_xgb = xgb_search.best_estimator_
    
    # Ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('lgbm', lgbm_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    
    # Training ensemble
    ensemble.fit(X_train_scaled, y_train)
    y_pred = ensemble.predict(X_test_scaled)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Printing results
    print("Ensemble Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (from XGBoost)
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
    
    return ensemble, scaler

def main():
    file_path = 'Hospital_Rating_Dataset.csv'  # Replace with actual file path
    df = load_and_preprocess_data(file_path)
    df = engineer_features(df)
    
    # Preparing features and target
    feature_cols = [col for col in df.columns if col not in ['facility_id', 'facility_name', 'Patient_Survey_Star_Rating']]
    X = df[feature_cols]
    y = df['Patient_Survey_Star_Rating'].dropna()
    
    # Aligning X and y
    valid_indices = y.index
    X = X.loc[valid_indices]
    
    # Preprocessing and feature selection
    X_selected, selected_features = preprocess_and_select_features(X, y)
    
    # Training and evaluating
    model, scaler = train_and_evaluate(X_selected, y, selected_features)
    
    # Saving model and scaler
    joblib.dump(model, 'hospital_rating_model_enhanced.pkl')
    joblib.dump(scaler, 'scaler_enhanced.pkl')
    joblib.dump(selected_features, 'selected_features.pkl')

if __name__ == '__main__':
    main()