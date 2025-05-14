import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Load the trained model and related files
@st.cache_resource
def load_model():
    model = joblib.load('hospital_rating_model_enhanced.pkl')
    scaler = joblib.load('scaler_enhanced.pkl')
    selected_features = joblib.load('selected_features.pkl')
    return model, scaler, selected_features

def predict_rating_from_inputs(inputs, model, scaler, selected_features):
    """
    Process inputs and make a prediction with proper feature engineering
    """
    # Create a base dataframe with all required features initialized to 0
    data = {feature: 0.0 for feature in selected_features}
    df = pd.DataFrame([data])
    
    # Update the dataframe with our inputs
    # Set direct user inputs
    for key, value in inputs.items():
        if key in selected_features:
            df[key] = value
    
    # Handle the "Always" percentage features (A_P)
    for feature in selected_features:
        if "A_P" in feature:
            if "NURSE" in feature:
                df[feature] = inputs.get('nurse_always_pct', inputs.get('avg_nurse_percent', 80))
            elif "DOCTOR" in feature:
                df[feature] = inputs.get('doctor_always_pct', inputs.get('avg_doctor_percent', 75))
            elif "BATH" in feature or "HELP" in feature:
                df[feature] = inputs.get('help_always_pct', 65)
            elif "SIDE_EFFECTS" in feature or "MED" in feature:
                df[feature] = inputs.get('med_always_pct', inputs.get('medication_pct', 70))
            elif "CALL_BUTTON" in feature:
                df[feature] = inputs.get('responsiveness_pct', 65)
            elif "COMP_1" in feature:  # Communication with Nurses
                df[feature] = inputs.get('avg_nurse_percent', 80)
            elif "COMP_2" in feature:  # Communication with Doctors
                df[feature] = inputs.get('avg_doctor_percent', 75)
            elif "COMP_3" in feature:  # Responsiveness of Hospital Staff
                df[feature] = inputs.get('responsiveness_pct', 65)
            elif "COMP_5" in feature:  # Communication About Medicines
                df[feature] = inputs.get('medication_pct', 70)
            else:
                # Use overall always percentage for other features
                df[feature] = inputs.get('avg_always_percent', 75)
    
    # Handle the "Sometimes/Never" percentage features (SN_P)
    for feature in selected_features:
        if "SN_P" in feature:
            related_a_feature = feature.replace("SN_P", "A_P")
            if related_a_feature in df.columns:
                # SN_P is roughly inverse of A_P (with adjustment for 3-point scale)
                a_value = df[related_a_feature].values[0]
                # Calculate SN value - assuming "Usually" takes remaining percentage not in Always or SN
                df[feature] = max(0, min(100 - a_value - 15, 30))  # Cap at reasonable values
            else:
                df[feature] = inputs.get('avg_sn_percent', 15)
    
    # Linear mean values - these are important for the model
    for feature in selected_features:
        if "LINEAR_SCORE" in feature:
            base_value = 3.0  # Default linear score (mid-range)
            
            # Adjust based on related percentage features
            if "COMP_1" in feature:  # Nurse communication
                base_value = inputs.get('avg_nurse_percent', 80) / 25  # Scale to ~1-4 range
            elif "COMP_2" in feature:  # Doctor communication
                base_value = inputs.get('avg_doctor_percent', 75) / 25  # Scale to ~1-4 range
            elif "COMP_3" in feature:  # Responsiveness
                base_value = inputs.get('responsiveness_pct', 65) / 25  # Scale to ~1-4 range
            elif "COMP_5" in feature:  # Medicine communication
                base_value = inputs.get('medication_pct', 70) / 25  # Scale to ~1-4 range
            elif "HSP_RATING" in feature:  # Hospital rating
                # This is especially important, as it directly relates to the target
                # Use a strong combination of factors to estimate this
                nurse_weight = 0.35
                doctor_weight = 0.3
                cleanliness_weight = 0.2
                responsive_weight = 0.15
                
                nurse_factor = inputs.get('avg_nurse_percent', 80) / 100
                doctor_factor = inputs.get('avg_doctor_percent', 75) / 100
                cleanliness_factor = inputs.get('cleanliness', 70) / 100
                responsive_factor = inputs.get('responsiveness_pct', 65) / 100
                
                weighted_score = (nurse_weight * nurse_factor + 
                                 doctor_weight * doctor_factor + 
                                 cleanliness_weight * cleanliness_factor +
                                 responsive_weight * responsive_factor)
                
                # Scale to typical range for this metric (1-4)
                base_value = 1 + weighted_score * 3
            
            df[feature] = min(4.0, max(1.0, base_value))  # Keep in reasonable range
    
    # Handle special calculated features
    if 'avg_always_percent' in selected_features:
        df['avg_always_percent'] = inputs.get('avg_always_percent', 75)
    
    if 'max_always_percent' in selected_features:
        # Usually about 10-15% higher than the average
        df['max_always_percent'] = min(100, inputs.get('avg_always_percent', 75) * 1.15)
    
    if 'avg_sn_percent' in selected_features:
        df['avg_sn_percent'] = inputs.get('avg_sn_percent', 15)
    
    if 'weighted_rating' in selected_features:
        # This is critical for prediction: it's the product of estimated rating and response rate
        # Estimate the rating first based on key metrics
        nurse_pct = inputs.get('avg_nurse_percent', 80)
        doctor_pct = inputs.get('avg_doctor_percent', 75)
        sn_pct = inputs.get('avg_sn_percent', 15)
        
        # The response rate is important for this weight
        response_rate = inputs.get('response_rate', 30)
        
        # A better way to estimate the star rating (0-4 scale for raw prediction)
        if nurse_pct >= 90 and doctor_pct >= 90 and sn_pct <= 5:
            estimated_rating = 4  # 5-star outcome
        elif nurse_pct >= 80 and doctor_pct >= 80 and sn_pct <= 10:
            estimated_rating = 3  # 4-star outcome
        elif nurse_pct >= 70 and doctor_pct >= 70 and sn_pct <= 15:
            estimated_rating = 2  # 3-star outcome
        elif nurse_pct >= 60 and doctor_pct >= 60:
            estimated_rating = 1  # 2-star outcome
        else:
            estimated_rating = 0  # 1-star outcome
        
        # Calculate weighted rating
        df['weighted_rating'] = estimated_rating * response_rate / 100
    
    # Make sure we have the right features and in the right order
    final_features = df[selected_features]
    
    # Scale features using the same scaler used during training
    X_scaled = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    prediction_proba = None
    
    # Get probabilities if available
    try:
        prediction_proba = model.predict_proba(X_scaled)[0]
    except:
        # Not all models support predict_proba
        pass
    
    # Convert from 0-indexed to 1-5 star rating
    star_rating = prediction + 1
    
    return star_rating, prediction_proba, final_features

def main():
    st.title("Hospital Rating Predictor")
    st.write("""
    ## Predict Hospital Ratings Based on Survey Responses
    This app uses machine learning to predict a hospital's star rating based on patient survey data.
    """)
    
    # Load the model
    model, scaler, selected_features = load_model()
    
    # Create input form
    st.sidebar.header("Enter Hospital Survey Data")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Basic Metrics", "Advanced Metrics"])
    
    with tab1:
        st.header("Basic Survey Metrics")
        
        # Always responses (most important feature from our model)
        avg_always_percent = st.slider(
            "Average 'Always' Response Percentage", 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0,
            help="Average percentage of patients who answered 'Always' to questions about hospital quality"
        )
        
        # Sometimes/Never responses (another top feature)
        avg_sn_percent = st.slider(
            "Average 'Sometimes/Never' Response Percentage", 
            min_value=0.0, 
            max_value=100.0, 
            value=10.0,
            help="Average percentage of patients who answered 'Sometimes' or 'Never' to questions about hospital quality"
        )
        
        # Number of completed surveys
        num_surveys = st.slider(
            "Number of Completed Surveys", 
            min_value=50, 
            max_value=1000, 
            value=300,
            help="Total number of completed patient surveys"
        )
        
        # Survey response rate
        response_rate = st.slider(
            "Survey Response Rate (%)", 
            min_value=5.0, 
            max_value=100.0, 
            value=30.0,
            help="Percentage of patients who responded to the survey request"
        )
    
    with tab2:
        st.header("Category-Specific Metrics")
        
        # Nurse communication
        nurse_percent = st.slider(
            "Nurse Communication Rating (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=80.0,
            help="Percentage of positive responses regarding nurse communication"
        )
        
        # Doctor communication
        doctor_percent = st.slider(
            "Doctor Communication Rating (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=75.0,
            help="Percentage of positive responses regarding doctor communication"
        )
        
        # Hospital cleanliness
        cleanliness = st.slider(
            "Hospital Cleanliness Rating (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=70.0,
            help="Percentage of positive responses regarding hospital cleanliness"
        )
        
        # Medication explanation
        medication_pct = st.slider(
            "Medication Explanation Rating (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=65.0,
            help="Percentage of positive responses regarding medication explanation"
        )
        
        # Responsiveness of staff
        responsiveness_pct = st.slider(
            "Staff Responsiveness Rating (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=65.0,
            help="Percentage of positive responses regarding staff responsiveness"
        )
    
    # Button to make prediction
    if st.button("Predict Hospital Rating"):
        # Collect all inputs into a single dictionary
        inputs = {
            'avg_always_percent': avg_always_percent,
            'avg_sn_percent': avg_sn_percent,
            'Number_of_Completed_Surveys': num_surveys,
            'Survey_Response_Rate_Percent': response_rate,
            'response_rate': response_rate,
            'avg_nurse_percent': nurse_percent,
            'avg_doctor_percent': doctor_percent,
            'cleanliness': cleanliness,
            'medication_pct': medication_pct,
            'responsiveness_pct': responsiveness_pct
        }
        
        # Get prediction
        star_rating, prediction_proba, features_used = predict_rating_from_inputs(
            inputs, model, scaler, selected_features
        )
        
        # Display prediction
        st.success(f"The predicted hospital rating is: {star_rating:.0f} stars")
        
        # Display a visual representation of the rating
        st.write("⭐" * int(star_rating))
        
        # Display confidence and interpretation
        st.subheader("What this means:")
        if star_rating >= 4:
            st.write("This hospital is likely to receive excellent reviews from patients, reflecting high-quality care and patient satisfaction.")
        elif star_rating >= 3:
            st.write("This hospital is likely to receive good reviews, with patients generally satisfied with their care.")
        else:
            st.write("This hospital may need to improve certain aspects of patient care and experience.")
        
        # Show top factors that would improve the rating
        st.subheader("How to improve this rating:")
        if star_rating < 5:
            st.write("To achieve a higher rating, focus on improving:")
            
            if avg_always_percent < 90:
                st.write(f"1. Increase 'Always' positive responses (currently {avg_always_percent:.1f}%, aim for >90%)")
            
            if avg_sn_percent > 5:
                st.write(f"2. Reduce 'Sometimes/Never' negative responses (currently {avg_sn_percent:.1f}%, aim for <5%)")
            
            if nurse_percent < 90:
                st.write(f"3. Improve nurse communication (currently {nurse_percent:.1f}%, aim for >90%)")
            
            if doctor_percent < 90:
                st.write(f"4. Enhance doctor communication (currently {doctor_percent:.1f}%, aim for >90%)")
            
            if response_rate < 40:
                st.write(f"5. Increase survey response rate (currently {response_rate:.1f}%, aim for >40%)")
        else:
            st.write("Excellent! This hospital is already predicted to achieve the highest rating.")
        
        # Show the most important feature values used
        with st.expander("View important feature values used in prediction"):
            st.write("Key features and their values:")
            st.write(f"- Average 'Always' Response: {avg_always_percent:.1f}%")
            st.write(f"- Average 'Sometimes/Never' Response: {avg_sn_percent:.1f}%")
            st.write(f"- Nurse Communication: {nurse_percent:.1f}%")
            st.write(f"- Doctor Communication: {doctor_percent:.1f}%")
            st.write(f"- Staff Responsiveness: {responsiveness_pct:.1f}%")
            st.write(f"- Medication Communication: {medication_pct:.1f}%")
            st.write(f"- Cleanliness: {cleanliness:.1f}%")
            st.write(f"- Survey Response Rate: {response_rate:.1f}%")
            st.write(f"- Number of Completed Surveys: {num_surveys}")

if __name__ == "__main__":
    main()
