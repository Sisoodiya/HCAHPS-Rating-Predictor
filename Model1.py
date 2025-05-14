import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import uuid
import warnings
warnings.filterwarnings('ignore')


# Loading and preprocessing the dataset
def load_and_preprocess_data(file_path):
    # Reading the CSV file
    df = pd.read_csv(file_path)
    
    # Replacing 'Not Applicable' with NaN
    df.replace('Not Applicable', np.nan, inplace=True)
    
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
    ).reset_index()
    
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

# Feature engineering
def engineer_features(df):
    # Creating aggregated features
    always_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'A_P' in col]
    df['avg_always_percent'] = df[always_cols].mean(axis=1)
    
    # Normalizing survey volume
    df['normalized_surveys'] = df['Number_of_Completed_Surveys'] / df['Number_of_Completed_Surveys'].max()    
    return df

# Training and evaluating the model
def train_and_evaluate(X, y):
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initializing models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    
    # Training and evaluating Random Forest
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    # Training and evaluating XGBoost
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    
    # Cross-validation for XGBoost
    xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # Printing results
    print("Random Forest Results:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")
    print(f"MAE: {rf_mae:.4f}")
    print("\nXGBoost Results:")
    print(f"Accuracy: {xgb_accuracy:.4f}")
    print(f"F1 Score: {xgb_f1:.4f}")
    print(f"MAE: {xgb_mae:.4f}")
    print(f"Cross-Validation Accuracy (XGBoost): {xgb_cv_scores.mean():.4f} (+/- {xgb_cv_scores.std() * 2:.4f})")
    
    # Feature importance (XGBoost)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 5 Important Features (XGBoost):")
    print(feature_importance.head())
    
    return xgb_model, scaler

def main():
    file_path = 'Hospital_Rating_Dataset.csv' # Replace with actual file path
    df = load_and_preprocess_data(file_path)
    df = engineer_features(df)
    
    # Preparing features and target
    feature_cols = [col for col in df.columns if col not in ['facility_id', 'facility_name', 'Patient_Survey_Star_Rating']]
    X = df[feature_cols].fillna(0)  # Filling NaNs with 0 for simplicity
    y = df['Patient_Survey_Star_Rating'].dropna()
    
    # Aligning X and y
    valid_indices = y.index
    X = X.loc[valid_indices]
    
    # Training and evaluating
    model, scaler = train_and_evaluate(X, y)
    
    # Saving model (optional)
    import joblib
    joblib.dump(model, 'hospital_rating_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
      # Replace with actual file path
    main()