import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the hospital rating dataset"""
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
    
    # Adjusting target to be 0-indexed for model compatibility
    target['Patient_Survey_Star_Rating'] = target['Patient_Survey_Star_Rating'] - 1
    
    # Merging pivoted features with target
    final_df = pivot_df.merge(target, on='facility_id')
    
    # Adding survey counts and response rate as features
    survey_cols = df[['facility_id', 'Number_of_Completed_Surveys', 'Survey_Response_Rate_Percent']].groupby('facility_id').mean()
    final_df = final_df.merge(survey_cols, on='facility_id')
    
    return final_df

def engineer_features(df):
    """Create engineered features for improved model performance"""
    # Always responses percentage
    always_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'A_P' in col]
    df['avg_always_percent'] = df[always_cols].mean(axis=1)
    df['max_always_percent'] = df[always_cols].max(axis=1)
    
    # Sometimes/Never responses percentage
    sn_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'SN_P' in col]
    df['avg_sn_percent'] = df[sn_cols].mean(axis=1)
    
    # Nurse and doctor communication
    nurse_cols = [col for col in df.columns if 'NURSE' in col and 'A_P' in col]
    doctor_cols = [col for col in df.columns if 'DOCTOR' in col and 'A_P' in col]
    
    if nurse_cols:
        df['avg_nurse_percent'] = df[nurse_cols].mean(axis=1)
    if doctor_cols:
        df['avg_doctor_percent'] = df[doctor_cols].mean(axis=1)
    
    # Interaction between nurse and doctor communication
    if 'avg_nurse_percent' in df.columns and 'avg_doctor_percent' in df.columns:
        df['nurse_doctor_interaction'] = df['avg_nurse_percent'] * df['avg_doctor_percent']
    
    # Ratio of Always to Sometimes/Never responses
    df['always_to_sn_ratio'] = df['avg_always_percent'] / (df['avg_sn_percent'] + 1e-5)
    
    # Normalizing survey volume
    df['normalized_surveys'] = df['Number_of_Completed_Surveys'] / df['Number_of_Completed_Surveys'].max()
    
    # Create a weighted rating feature (estimated rating * response rate)
    linear_cols = [col for col in df.columns if 'LINEAR_SCORE' in col]
    if linear_cols:
        df['avg_linear_score'] = df[linear_cols].mean(axis=1)
        df['weighted_rating'] = df['avg_linear_score'] * df['Survey_Response_Rate_Percent'] / 100
    
    return df

def build_cnn_model(input_shape, num_classes=5):
    """Build a deep neural network model for hospital rating prediction"""
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(X, y):
    """Train and evaluate the CNN model for hospital rating prediction"""
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    input_shape = X_train_scaled.shape[1]
    model = build_cnn_model(input_shape)
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=-1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Print classification report
    report = classification_report(y_test, y_pred_classes)
    print("Classification Report:")
    print(report)
    
    # Select important features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.std(X_train_scaled, axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    selected_features = feature_importance['feature'].tolist()
    
    return model, scaler, selected_features

def save_model(model, scaler, selected_features, model_path='Models/hospital_rating_model_cnn.keras', scaler_path='Scalars/scaler_cnn.pkl', features_path='Scalars/selected_features_cnn.pkl'):
    """Save the trained model, scaler, and selected features"""
    # Save Keras model in the recommended native format
    model.save(model_path)
    
    # Save scaler and selected features using joblib
    joblib.dump(scaler, scaler_path)
    joblib.dump(selected_features, features_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Selected features saved to {features_path}")

def main():
    """Main function to train and save the CNN model"""
    print("Loading and preprocessing data...")
    file_path = 'Hospital_Rating_Dataset.csv'  # Replace with actual file path
    df = load_and_preprocess_data(file_path)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    # Preparing features and target
    feature_cols = [col for col in df.columns if col not in ['facility_id', 'facility_name', 'Patient_Survey_Star_Rating']]
    X = df[feature_cols].fillna(0)  # Filling NaNs with 0 for simplicity
    y = df['Patient_Survey_Star_Rating'].dropna()
    
    # Aligning X and y
    valid_indices = y.index
    X = X.loc[valid_indices]
    
    print(f"Training with {X.shape[1]} features and {len(y)} samples...")
    model, scaler, selected_features = train_and_evaluate(X, y)
    
    # Save model and related files
    save_model(model, scaler, selected_features)

if __name__ == '__main__':
    main()
