import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
import io
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Hospital Rating System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #0047AB;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #4682B4;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #1E90FF;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .prediction-result {
        background-color: #f0fff0;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .hcahps-question {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .rating-card {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        text-align: center;
    }
    .recommendation-card {
        background-color: #fff0f0;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .stRadio > div {
        display: flex;
        flex-direction: row;
    }
    .stRadio label {
        margin-right: 15px;
        font-weight: normal;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and related files
@st.cache_resource
def load_model():
    model_type = get_selected_model()
    
    try:
        if model_type == 'enhanced':
            model = joblib.load('hospital_rating_model_enhanced.pkl')
            scaler = joblib.load('scaler_enhanced.pkl')
            selected_features = joblib.load('selected_features.pkl')
        else:
            # Use standard model
            model = joblib.load('hospital_rating_model.pkl')
            scaler = joblib.load('scaler.pkl')
            selected_features = joblib.load('selected_features.pkl')
        
        return model, scaler, selected_features
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Initialize session state for storing patient data
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        'patient_id': '',
        'facility_id': '',
        'facility_name': '',
        'admission_date': None,
        'discharge_date': None,
        'survey_responses': {},
        'predictions': {}
    }

def calculate_section_scores(responses, section):
    """Calculate average scores for a specific section of questions"""
    section_questions = [q for q in responses if q.startswith(section)]
    if not section_questions:
        return 0
    
    # Map responses to numeric values (1-4 scale where 4 is best)
    mapping = {
        'Never': 1,
        'Sometimes': 2,
        'Usually': 3,
        'Always': 4,
        'Definitely no': 1,
        'Probably no': 2,
        'Probably yes': 3,
        'Definitely yes': 4,
        'Strongly disagree': 1,
        'Disagree': 2,
        'Agree': 3,
        'Strongly agree': 4,
        '0-6': 1,
        '7-8': 3,
        '9-10': 4,
        'No': 1,
        'Yes': 4
    }
    
    # Calculate score
    values = [mapping.get(responses[q], 0) for q in section_questions if responses.get(q)]
    if not values:
        return 0
    
    # Calculate percentage (0-100 scale where 100 is best)
    avg_score = sum(values) / len(values)
    percentage = ((avg_score - 1) / 3) * 100
    
    return percentage

def map_responses_to_model_features(responses):
    """Convert survey responses to features expected by the model"""
    features = {}
    
    # Map survey questions to HCAHPS model features
    # Nurse communication
    nurse_always = 0
    nurse_sometimes_never = 0
    nurse_qs = ['nurse_communication_1', 'nurse_communication_2', 'nurse_communication_3']
    nurse_responses = [responses.get(q) for q in nurse_qs if q in responses]
    
    if nurse_responses:
        nurse_always = sum(1 for r in nurse_responses if r == 'Always') / len(nurse_responses) * 100
        nurse_sometimes_never = sum(1 for r in nurse_responses if r in ['Sometimes', 'Never']) / len(nurse_responses) * 100
        
    features['HCAHPS_Answer_Percent_H_NURSE_EXPLAIN_A_P'] = nurse_always
    features['HCAHPS_Answer_Percent_H_NURSE_LISTEN_A_P'] = nurse_always
    features['HCAHPS_Answer_Percent_H_NURSE_RESPECT_A_P'] = nurse_always
    features['HCAHPS_Answer_Percent_H_NURSE_EXPLAIN_SN_P'] = nurse_sometimes_never
    features['HCAHPS_Answer_Percent_H_NURSE_LISTEN_SN_P'] = nurse_sometimes_never
    features['HCAHPS_Answer_Percent_H_NURSE_RESPECT_SN_P'] = nurse_sometimes_never
    
    # Doctor communication
    doctor_always = 0
    doctor_sometimes_never = 0
    doctor_qs = ['doctor_communication_1', 'doctor_communication_2', 'doctor_communication_3']
    doctor_responses = [responses.get(q) for q in doctor_qs if q in responses]
    
    if doctor_responses:
        doctor_always = sum(1 for r in doctor_responses if r == 'Always') / len(doctor_responses) * 100
        doctor_sometimes_never = sum(1 for r in doctor_responses if r in ['Sometimes', 'Never']) / len(doctor_responses) * 100
    
    features['HCAHPS_Answer_Percent_H_DOCTOR_EXPLAIN_A_P'] = doctor_always
    features['HCAHPS_Answer_Percent_H_DOCTOR_LISTEN_A_P'] = doctor_always
    features['HCAHPS_Answer_Percent_H_DOCTOR_RESPECT_A_P'] = doctor_always
    features['HCAHPS_Answer_Percent_H_DOCTOR_EXPLAIN_SN_P'] = doctor_sometimes_never
    features['HCAHPS_Answer_Percent_H_DOCTOR_LISTEN_SN_P'] = doctor_sometimes_never
    
    # Staff responsiveness
    resp_always = 0
    resp_sometimes_never = 0
    resp_qs = ['hospital_experience_1', 'hospital_experience_2']
    resp_responses = [responses.get(q) for q in resp_qs if q in responses]
    
    if resp_responses:
        resp_always = sum(1 for r in resp_responses if r == 'Always') / len(resp_responses) * 100
        resp_sometimes_never = sum(1 for r in resp_responses if r in ['Sometimes', 'Never']) / len(resp_responses) * 100
    
    features['HCAHPS_Answer_Percent_H_CALL_BUTTON_A_P'] = resp_always
    features['HCAHPS_Answer_Percent_H_BATH_HELP_A_P'] = resp_always
    features['HCAHPS_Answer_Percent_H_CALL_BUTTON_SN_P'] = resp_sometimes_never
    features['HCAHPS_Answer_Percent_H_BATH_HELP_SN_P'] = resp_sometimes_never
    
    # Hospital environment
    clean_always = 0
    clean_sometimes_never = 0
    if 'hospital_environment_1' in responses:
        clean_always = 100 if responses['hospital_environment_1'] == 'Always' else 0
        clean_sometimes_never = 100 if responses['hospital_environment_1'] in ['Sometimes', 'Never'] else 0
    
    features['HCAHPS_Answer_Percent_H_COMP_1_A_P'] = clean_always
    features['HCAHPS_Answer_Percent_H_COMP_1_SN_P'] = clean_sometimes_never
    
    quiet_always = 0
    quiet_sometimes_never = 0
    if 'hospital_environment_2' in responses:
        quiet_always = 100 if responses['hospital_environment_2'] == 'Always' else 0
        quiet_sometimes_never = 100 if responses['hospital_environment_2'] in ['Sometimes', 'Never'] else 0
    
    features['HCAHPS_Answer_Percent_H_COMP_2_A_P'] = quiet_always
    features['HCAHPS_Answer_Percent_H_COMP_2_SN_P'] = quiet_sometimes_never
    
    # Medication communication
    med_always = 0
    med_sometimes_never = 0
    med_qs = ['hospital_experience_3', 'hospital_experience_4']
    med_responses = [responses.get(q) for q in med_qs if q in responses]
    
    if med_responses:
        med_always = sum(1 for r in med_responses if r == 'Always') / len(med_responses) * 100
        med_sometimes_never = sum(1 for r in med_responses if r in ['Sometimes', 'Never']) / len(med_responses) * 100
    
    features['HCAHPS_Answer_Percent_H_MED_FOR_A_P'] = med_always
    features['HCAHPS_Answer_Percent_H_SIDE_EFFECTS_A_P'] = med_always
    features['HCAHPS_Answer_Percent_H_MED_FOR_SN_P'] = med_sometimes_never
    features['HCAHPS_Answer_Percent_H_SIDE_EFFECTS_SN_P'] = med_sometimes_never
    
    # Discharge information
    discharge_yes = 0
    discharge_no = 0
    if 'discharge_1' in responses and 'discharge_2' in responses:
        discharge_yes = sum(1 for r in [responses['discharge_1'], responses['discharge_2']] if r == 'Yes') / 2 * 100
        discharge_no = sum(1 for r in [responses['discharge_1'], responses['discharge_2']] if r == 'No') / 2 * 100
    
    features['HCAHPS_Answer_Percent_H_COMP_6_Y_P'] = discharge_yes
    features['HCAHPS_Answer_Percent_H_COMP_6_N_P'] = discharge_no
    
    # Care transition
    ct_agree = 0
    ct_disagree = 0
    ct_qs = ['care_transition_1', 'care_transition_2', 'care_transition_3']
    ct_responses = [responses.get(q) for q in ct_qs if q in responses]
    
    if ct_responses:
        ct_agree = sum(1 for r in ct_responses if r in ['Agree', 'Strongly agree']) / len(ct_responses) * 100
        ct_disagree = sum(1 for r in ct_responses if r in ['Disagree', 'Strongly disagree']) / len(ct_responses) * 100
    
    features['HCAHPS_Answer_Percent_H_CT_PREFER_SA'] = ct_agree
    features['HCAHPS_Answer_Percent_H_CT_UNDER_SA'] = ct_agree
    features['HCAHPS_Answer_Percent_H_COMP_7_SA'] = ct_agree
    features['HCAHPS_Answer_Percent_H_CT_PREFER_D_SD'] = ct_disagree
    features['HCAHPS_Answer_Percent_H_CT_UNDER_D_SD'] = ct_disagree
    features['HCAHPS_Answer_Percent_H_COMP_7_D_SD'] = ct_disagree
    
    # Overall hospital rating
    hospital_rating_0_6 = 0
    hospital_rating_9_10 = 0
    if 'overall_rating' in responses:
        rating = responses['overall_rating']
        hospital_rating_0_6 = 100 if rating == '0-6' else 0
        hospital_rating_9_10 = 100 if rating == '9-10' else 0
    
    features['HCAHPS_Answer_Percent_H_HSP_RATING_0_6'] = hospital_rating_0_6
    features['HCAHPS_Answer_Percent_H_HSP_RATING_9_10'] = hospital_rating_9_10
    
    # Recommendation
    recommend_dn = 0
    if 'recommend_hospital' in responses:
        recommend_dn = 100 if responses['recommend_hospital'] == 'Definitely no' else 0
    
    features['HCAHPS_Answer_Percent_H_RECMND_DN'] = recommend_dn
    
    # Linear mean values (approximations based on survey responses)
    nurse_score = calculate_section_scores(responses, 'nurse_communication')
    doctor_score = calculate_section_scores(responses, 'doctor_communication')
    hospital_env_score = calculate_section_scores(responses, 'hospital_environment')
    
    # Map scores to linear values (1-4 scale)
    features['HCAHPS_Linear_Mean_Value_H_COMP_1_LINEAR_SCORE'] = 1 + (hospital_env_score / 100 * 3)
    features['HCAHPS_Linear_Mean_Value_H_COMP_2_LINEAR_SCORE'] = 1 + (hospital_env_score / 100 * 3)
    features['HCAHPS_Linear_Mean_Value_H_COMP_3_LINEAR_SCORE'] = 1 + (nurse_score / 100 * 3)
    features['HCAHPS_Linear_Mean_Value_H_COMP_5_LINEAR_SCORE'] = 1 + (doctor_score / 100 * 3)
    features['HCAHPS_Linear_Mean_Value_H_COMP_6_LINEAR_SCORE'] = 1 + (calculate_section_scores(responses, 'discharge') / 100 * 3)
    features['HCAHPS_Linear_Mean_Value_H_COMP_7_LINEAR_SCORE'] = 1 + (calculate_section_scores(responses, 'care_transition') / 100 * 3)
    
    # Estimate overall scores
    overall_rating_score = 0
    if 'overall_rating' in responses:
        rating_map = {'0-6': 1, '7-8': 3, '9-10': 4}
        overall_rating_score = rating_map.get(responses['overall_rating'], 0)
    
    features['HCAHPS_Linear_Mean_Value_H_HSP_RATING_LINEAR_SCORE'] = overall_rating_score
    
    recommend_score = 0
    if 'recommend_hospital' in responses:
        recommend_map = {'Definitely no': 1, 'Probably no': 2, 'Probably yes': 3, 'Definitely yes': 4}
        recommend_score = recommend_map.get(responses['recommend_hospital'], 0)
    
    features['HCAHPS_Linear_Mean_Value_H_RECMND_LINEAR_SCORE'] = recommend_score
    
    # Calculate aggregate metrics
    always_percentages = [v for k, v in features.items() if '_A_P' in k]
    sn_percentages = [v for k, v in features.items() if '_SN_P' in k]
    
    # Aggregate features
    if always_percentages:
        features['avg_always_percent'] = sum(always_percentages) / len(always_percentages)
        features['max_always_percent'] = max(always_percentages) if always_percentages else 0
    else:
        features['avg_always_percent'] = 0
        features['max_always_percent'] = 0
    
    if sn_percentages:
        features['avg_sn_percent'] = sum(sn_percentages) / len(sn_percentages)
    else:
        features['avg_sn_percent'] = 0
    
    # Weighted rating (estimated)
    estimated_rating = 0
    if nurse_score >= 90 and doctor_score >= 90:
        estimated_rating = 4  # 5-star outcome
    elif nurse_score >= 80 and doctor_score >= 80:
        estimated_rating = 3  # 4-star outcome
    elif nurse_score >= 70 and doctor_score >= 70:
        estimated_rating = 2  # 3-star outcome
    elif nurse_score >= 60 and doctor_score >= 60:
        estimated_rating = 1  # 2-star outcome
    
    features['weighted_rating'] = estimated_rating
    
    return features

def predict_rating(responses, model, scaler, selected_features):
    """Predict hospital rating based on survey responses"""
    # Check if we have enough responses
    if len(responses) < 5:
        raise ValueError("Not enough survey responses to make a prediction. Please complete at least 5 survey questions.")
    
    # Map responses to model features
    features = map_responses_to_model_features(responses)
    
    # Check if we have enough valid features
    valid_features_count = sum(1 for feature in selected_features if feature in features)
    min_required_features = len(selected_features) * 0.6  # At least 60% of features should be available
    
    if valid_features_count < min_required_features:
        raise ValueError(f"Insufficient data for accurate prediction. Please complete more survey questions (needed: {min_required_features:.0f}, got: {valid_features_count})")
    
    try:
        # Create feature vector with all required features
        X = pd.DataFrame([{feature: features.get(feature, 0.0) for feature in selected_features}])
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Get probabilities if available
        prediction_proba = None
        try:
            prediction_proba = model.predict_proba(X_scaled)[0]
        except:
            pass
        
        # Convert from 0-indexed to 1-5 star rating
        star_rating = prediction + 1
        
        # Ensure the rating is within valid range (1-5 stars)
        star_rating = max(1, min(5, star_rating))
        
        return star_rating, prediction_proba, features
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        raise ValueError(f"Failed to calculate hospital rating: {str(e)}")

# Function for model selection and prediction
def get_selected_model():
    """Get the currently selected model from session state"""
    if 'model_type' not in st.session_state:
        st.session_state.model_type = 'standard'
    return st.session_state.model_type

def generate_radar_chart(features):
    """Generate a radar chart for hospital metrics visualization"""
    # Extract key metrics
    categories = [
        'Nurse Communication', 
        'Doctor Communication', 
        'Staff Responsiveness', 
        'Hospital Cleanliness',
        'Quietness', 
        'Medication Communication',
        'Discharge Information',
        'Care Transition'
    ]
    
    # Calculate percentages for each category (0-100 scale)
    nurse_pct = features.get('avg_always_percent', 0)
    doctor_pct = features.get('HCAHPS_Answer_Percent_H_DOCTOR_EXPLAIN_A_P', 0)
    resp_pct = features.get('HCAHPS_Answer_Percent_H_CALL_BUTTON_A_P', 0)
    clean_pct = features.get('HCAHPS_Answer_Percent_H_COMP_1_A_P', 0)
    quiet_pct = features.get('HCAHPS_Answer_Percent_H_COMP_2_A_P', 0)
    med_pct = features.get('HCAHPS_Answer_Percent_H_MED_FOR_A_P', 0)
    discharge_pct = features.get('HCAHPS_Answer_Percent_H_COMP_6_Y_P', 0)
    care_pct = features.get('HCAHPS_Answer_Percent_H_CT_PREFER_SA', 0)
    
    values = [
        nurse_pct,
        doctor_pct,
        resp_pct,
        clean_pct,
        quiet_pct,
        med_pct,
        discharge_pct,
        care_pct
    ]
    
    # Close the circle by appending the first value to the end
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    # Create the radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(70, 130, 180, 0.5)',
        line=dict(color='rgb(70, 130, 180)'),
        name='Patient Responses'
    ))
    
    # Add benchmark line (90% is typically excellent)
    fig.add_trace(go.Scatterpolar(
        r=[90, 90, 90, 90, 90, 90, 90, 90, 90],
        theta=categories,
        line=dict(color='rgba(255, 99, 71, 0.8)', dash='dash'),
        name='Excellence Benchmark'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Hospital Performance Metrics"
    )
    
    return fig

def get_improvement_recommendations(features):
    """Generate recommendations based on survey responses"""
    recommendations = []
    
    # Nurse communication
    nurse_pct = features.get('avg_always_percent', 0)
    if nurse_pct < 80:
        recommendations.append({
            'category': 'Nurse Communication',
            'score': f"{nurse_pct:.1f}%",
            'tips': [
                "Implement hourly nurse rounding protocols",
                "Provide communication skills training for nursing staff",
                "Establish bedside shift reporting to involve patients"
            ]
        })
    
    # Doctor communication
    doctor_pct = features.get('HCAHPS_Answer_Percent_H_DOCTOR_EXPLAIN_A_P', 0)
    if doctor_pct < 80:
        recommendations.append({
            'category': 'Doctor Communication',
            'score': f"{doctor_pct:.1f}%",
            'tips': [
                "Implement AIDET (Acknowledge, Introduce, Duration, Explanation, Thank you) protocol",
                "Schedule dedicated time for doctor-patient discussions",
                "Provide communication skills workshops for physicians"
            ]
        })
    
    # Hospital cleanliness
    clean_pct = features.get('HCAHPS_Answer_Percent_H_COMP_1_A_P', 0)
    if clean_pct < 80:
        recommendations.append({
            'category': 'Hospital Cleanliness',
            'score': f"{clean_pct:.1f}%",
            'tips': [
                "Implement visual cleanliness inspections with feedback",
                "Ensure regular cleaning of high-touch surfaces",
                "Train staff on proper cleaning protocols"
            ]
        })
    
    # Medication communication
    med_pct = features.get('HCAHPS_Answer_Percent_H_MED_FOR_A_P', 0)
    if med_pct < 80:
        recommendations.append({
            'category': 'Medication Communication',
            'score': f"{med_pct:.1f}%",
            'tips': [
                "Use teach-back method when explaining medications",
                "Provide written medication information in patient-friendly language",
                "Schedule medication education at optimal times for patient comprehension"
            ]
        })
    
    return recommendations

def patient_info_form():
    """Patient demographic and hospital stay information form with validation"""
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    
    # Initialize form validation errors dictionary
    form_errors = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input(
            "Patient ID*", 
            value=st.session_state.patient_data.get('patient_id', ''), 
            key="patient_id_input",
            help="Required field - Enter a unique identifier for the patient"
        )
        if not patient_id:
            form_errors['patient_id'] = "Patient ID is required"
            
        facility_id = st.text_input(
            "Hospital/Facility ID*", 
            value=st.session_state.patient_data.get('facility_id', ''), 
            key="facility_id_input",
            help="Required field - Enter the facility identifier"
        )
        if not facility_id:
            form_errors['facility_id'] = "Facility ID is required"
    
    with col2:
        facility_name = st.text_input(
            "Hospital/Facility Name*", 
            value=st.session_state.patient_data.get('facility_name', ''), 
            key="facility_name_input",
            help="Required field - Enter the name of the hospital or facility"
        )
        if not facility_name:
            form_errors['facility_name'] = "Facility name is required"
            
        patient_gender = st.selectbox(
            "Gender", 
            options=["", "Male", "Female", "Other", "Prefer not to say"], 
            index=0, 
            key="gender_input"
        )
    
    st.markdown('<div class="section-title">Hospital Stay Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Get default dates from session state if available
    default_admission = st.session_state.patient_data.get('admission_date')
    default_discharge = st.session_state.patient_data.get('discharge_date')
    
    with col1:
        admission_date = st.date_input(
            "Admission Date*", 
            value=default_admission,
            key="admission_date_input",
            help="Required field - Enter the date patient was admitted"
        )
        if not admission_date:
            form_errors['admission_date'] = "Admission date is required"
    
    with col2:
        discharge_date = st.date_input(
            "Discharge Date*", 
            value=default_discharge,
            key="discharge_date_input",
            help="Required field - Enter the date patient was discharged"
        )
        if not discharge_date:
            form_errors['discharge_date'] = "Discharge date is required"
    
    # Date validation logic
    if admission_date and discharge_date and admission_date > discharge_date:
        form_errors['date_range'] = "Discharge date cannot be before admission date"
    
    # Display any validation errors
    if form_errors:
        st.warning("Please fix the following errors:")
        for error in form_errors.values():
            st.error(error)
    
    # Save entered data to session state regardless of validation
    # This allows partial form completion to be saved
    st.session_state.patient_data['patient_id'] = patient_id
    st.session_state.patient_data['facility_id'] = facility_id
    st.session_state.patient_data['facility_name'] = facility_name
    if admission_date:
        st.session_state.patient_data['admission_date'] = admission_date
    if discharge_date:
        st.session_state.patient_data['discharge_date'] = discharge_date
    if patient_gender:
        st.session_state.patient_data['gender'] = patient_gender
    
    # Return form data and validation status
    return {
        'patient_id': patient_id,
        'facility_id': facility_id,
        'facility_name': facility_name,
        'admission_date': admission_date,
        'discharge_date': discharge_date,
        'gender': patient_gender,
        'is_valid': len(form_errors) == 0
    }

def hcahps_survey_form():
    """HCAHPS patient survey form"""
    # Initialize responses dictionary if not already in session state
    if 'survey_responses' not in st.session_state.patient_data:
        st.session_state.patient_data['survey_responses'] = {}
    
    responses = {}
    
    # Section 1: Your Care from Nurses
    st.markdown('<div class="section-title">Your Care from Nurses</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did nurses treat you with courtesy and respect?</div>', unsafe_allow_html=True)
        nurse_respect = st.radio(
            "During this hospital stay, how often did nurses treat you with courtesy and respect?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="nurse_communication_1"
        )
        responses['nurse_communication_1'] = nurse_respect
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did nurses listen carefully to you?</div>', unsafe_allow_html=True)
        nurse_listen = st.radio(
            "During this hospital stay, how often did nurses listen carefully to you?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="nurse_communication_2"
        )
        responses['nurse_communication_2'] = nurse_listen
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did nurses explain things in a way you could understand?</div>', unsafe_allow_html=True)
        nurse_explain = st.radio(
            "During this hospital stay, how often did nurses explain things in a way you could understand?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="nurse_communication_3"
        )
        responses['nurse_communication_3'] = nurse_explain
    
    # Section 2: Your Care from Doctors
    st.markdown('<div class="section-title">Your Care from Doctors</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did doctors treat you with courtesy and respect?</div>', unsafe_allow_html=True)
        doctor_respect = st.radio(
            "During this hospital stay, how often did doctors treat you with courtesy and respect?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="doctor_communication_1"
        )
        responses['doctor_communication_1'] = doctor_respect
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did doctors listen carefully to you?</div>', unsafe_allow_html=True)
        doctor_listen = st.radio(
            "During this hospital stay, how often did doctors listen carefully to you?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="doctor_communication_2"
        )
        responses['doctor_communication_2'] = doctor_listen
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did doctors explain things in a way you could understand?</div>', unsafe_allow_html=True)
        doctor_explain = st.radio(
            "During this hospital stay, how often did doctors explain things in a way you could understand?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="doctor_communication_3"
        )
        responses['doctor_communication_3'] = doctor_explain
    
    # Section 3: The Hospital Environment
    st.markdown('<div class="section-title">The Hospital Environment</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often were your room and bathroom kept clean?</div>', unsafe_allow_html=True)
        room_clean = st.radio(
            "During this hospital stay, how often were your room and bathroom kept clean?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="hospital_environment_1"
        )
        responses['hospital_environment_1'] = room_clean
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often was the area around your room quiet at night?</div>', unsafe_allow_html=True)
        room_quiet = st.radio(
            "During this hospital stay, how often was the area around your room quiet at night?",
            options=["", "Never", "Sometimes", "Usually", "Always"],
            index=0,
            label_visibility="collapsed",
            key="hospital_environment_2"
        )
        responses['hospital_environment_2'] = room_quiet
    
    # Section 4: Your Experiences in this Hospital
    st.markdown('<div class="section-title">Your Experiences in this Hospital</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did you get help as soon as you wanted it after pressing the call button?</div>', unsafe_allow_html=True)
        call_button = st.radio(
            "During this hospital stay, how often did you get help as soon as you wanted it after pressing the call button?",
            options=["", "Never", "Sometimes", "Usually", "Always", "I never pressed the call button"],
            index=0,
            label_visibility="collapsed",
            key="hospital_experience_1"
        )
        responses['hospital_experience_1'] = call_button
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, how often did you get help with bathing, dressing, or using the bathroom as soon as you wanted?</div>', unsafe_allow_html=True)
        bath_help = st.radio(
            "During this hospital stay, how often did you get help with bathing, dressing, or using the bathroom as soon as you wanted?",
            options=["", "Never", "Sometimes", "Usually", "Always", "I didn't need help"],
            index=0,
            label_visibility="collapsed",
            key="hospital_experience_2"
        )
        responses['hospital_experience_2'] = bath_help
    
    with st.container():
        st.markdown('<div class="hcahps-question">Before giving you any new medicine, how often did hospital staff tell you what the medicine was for?</div>', unsafe_allow_html=True)
        med_for = st.radio(
            "Before giving you any new medicine, how often did hospital staff tell you what the medicine was for?",
            options=["", "Never", "Sometimes", "Usually", "Always", "I wasn't given any medicine"],
            index=0,
            label_visibility="collapsed",
            key="hospital_experience_3"
        )
        responses['hospital_experience_3'] = med_for
    
    with st.container():
        st.markdown('<div class="hcahps-question">Before giving you any new medicine, how often did hospital staff describe possible side effects in a way you could understand?</div>', unsafe_allow_html=True)
        side_effects = st.radio(
            "Before giving you any new medicine, how often did hospital staff describe possible side effects in a way you could understand?",
            options=["", "Never", "Sometimes", "Usually", "Always", "I wasn't given any medicine"],
            index=0,
            label_visibility="collapsed",
            key="hospital_experience_4"
        )
        responses['hospital_experience_4'] = side_effects
    
    # Section 5: When You Left the Hospital
    st.markdown('<div class="section-title">When You Left the Hospital</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, did staff talk with you about whether you would have the help you needed when you left the hospital?</div>', unsafe_allow_html=True)
        discharge_help = st.radio(
            "During this hospital stay, did staff talk with you about whether you would have the help you needed when you left the hospital?",
            options=["", "Yes", "No"],
            index=0,
            label_visibility="collapsed",
            key="discharge_1"
        )
        responses['discharge_1'] = discharge_help
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, did you get information in writing about what symptoms or health problems to look out for after you left the hospital?</div>', unsafe_allow_html=True)
        discharge_info = st.radio(
            "During this hospital stay, did you get information in writing about what symptoms or health problems to look out for after you left the hospital?",
            options=["", "Yes", "No"],
            index=0,
            label_visibility="collapsed",
            key="discharge_2"
        )
        responses['discharge_2'] = discharge_info
    
    # Section 6: Understanding Your Care When You Left the Hospital
    st.markdown('<div class="section-title">Understanding Your Care When You Left the Hospital</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">During this hospital stay, staff took my preferences and those of my family or caregiver into account in deciding what my health care needs would be when I left.</div>', unsafe_allow_html=True)
        care_pref = st.radio(
            "During this hospital stay, staff took my preferences and those of my family or caregiver into account in deciding what my health care needs would be when I left.",
            options=["", "Strongly disagree", "Disagree", "Agree", "Strongly agree"],
            index=0,
            label_visibility="collapsed",
            key="care_transition_1"
        )
        responses['care_transition_1'] = care_pref
    
    with st.container():
        st.markdown('<div class="hcahps-question">When I left the hospital, I had a good understanding of the things I was responsible for in managing my health.</div>', unsafe_allow_html=True)
        understand_health = st.radio(
            "When I left the hospital, I had a good understanding of the things I was responsible for in managing my health.",
            options=["", "Strongly disagree", "Disagree", "Agree", "Strongly agree"],
            index=0,
            label_visibility="collapsed",
            key="care_transition_2"
        )
        responses['care_transition_2'] = understand_health
    
    with st.container():
        st.markdown('<div class="hcahps-question">When I left the hospital, I clearly understood the purpose for taking each of my medications.</div>', unsafe_allow_html=True)
        understand_meds = st.radio(
            "When I left the hospital, I clearly understood the purpose for taking each of my medications.",
            options=["", "Strongly disagree", "Disagree", "Agree", "Strongly agree", "I was not given any medication when I left the hospital"],
            index=0,
            label_visibility="collapsed",
            key="care_transition_3"
        )
        responses['care_transition_3'] = understand_meds
    
    # Section 7: Overall Hospital Rating
    st.markdown('<div class="section-title">Overall Hospital Rating</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="hcahps-question">Using any number from 0 to 10, where 0 is the worst hospital possible and 10 is the best hospital possible, what number would you use to rate this hospital during your stay?</div>', unsafe_allow_html=True)
        rating = st.select_slider(
            "Using any number from 0 to 10, where 0 is the worst hospital possible and 10 is the best hospital possible, what number would you use to rate this hospital during your stay?",
            options=["0 (Worst)", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10 (Best)"],
            value=None,
            key="overall_rating_slider"
        )
        
        # Convert rating to the format needed by model
        if rating:
            if rating in ["0 (Worst)", "1", "2", "3", "4", "5", "6"]:
                responses['overall_rating'] = '0-6'
            elif rating in ["7", "8"]:
                responses['overall_rating'] = '7-8'
            elif rating in ["9", "10 (Best)"]:
                responses['overall_rating'] = '9-10'
    
    with st.container():
        st.markdown('<div class="hcahps-question">Would you recommend this hospital to your friends and family?</div>', unsafe_allow_html=True)
        recommend = st.radio(
            "Would you recommend this hospital to your friends and family?",
            options=["", "Definitely no", "Probably no", "Probably yes", "Definitely yes"],
            index=0,
            label_visibility="collapsed",
            key="recommend_hospital"
        )
        responses['recommend_hospital'] = recommend
    
    # Save responses to session state
    if responses:
        st.session_state.patient_data['survey_responses'] = responses
    
    return responses

def display_section_score(title, score):
    """Display a section score with proper formatting"""
    st.markdown(f"""
    <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
        <span style="font-weight: bold;">{title}:</span> {score:.1f}%
    </div>
    """, unsafe_allow_html=True)

def save_patient_data_to_csv():
    """Generate a CSV file containing the patient data and offer it for download"""
    if not st.session_state.patient_data['patient_id']:
        st.error("Patient ID is required.")
        return None
    
    # Prepare data for CSV
    patient_data = st.session_state.patient_data
    
    # Create a DataFrame for patient info
    patient_info = {
        'patient_id': [patient_data.get('patient_id', '')],
        'facility_id': [patient_data.get('facility_id', '')],
        'facility_name': [patient_data.get('facility_name', '')],
        'admission_date': [patient_data.get('admission_date', '')],
        'discharge_date': [patient_data.get('discharge_date', '')],
        'gender': [patient_data.get('gender', '')],
    }
    
    # Add survey responses
    survey_responses = patient_data.get('survey_responses', {})
    for question, response in survey_responses.items():
        patient_info[f"response_{question}"] = [response]
    
    # Add predictions
    predictions = patient_data.get('predictions', {})
    if predictions:
        patient_info['star_rating'] = [predictions.get('star_rating', '')]
        patient_info['prediction_timestamp'] = [predictions.get('timestamp', '')]
    
    # Create DataFrame
    df = pd.DataFrame(patient_info)
    
    # Generate CSV
    csv = df.to_csv(index=False)
    
    # Encode as base64 for download
    b64 = base64.b64encode(csv.encode()).decode()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patient_{patient_data['patient_id']}_{timestamp}.csv"
    
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Patient Data</a>'
    
    return href

def load_patient_data_from_csv(uploaded_file):
    """Load patient data from an uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Extract patient info
        patient_data = {
            'patient_id': df.iloc[0]['patient_id'] if 'patient_id' in df.columns else '',
            'facility_id': df.iloc[0]['facility_id'] if 'facility_id' in df.columns else '',
            'facility_name': df.iloc[0]['facility_name'] if 'facility_name' in df.columns else '',
        }
        
        if 'admission_date' in df.columns and pd.notna(df.iloc[0]['admission_date']):
            patient_data['admission_date'] = pd.to_datetime(df.iloc[0]['admission_date']).date()
        
        if 'discharge_date' in df.columns and pd.notna(df.iloc[0]['discharge_date']):
            patient_data['discharge_date'] = pd.to_datetime(df.iloc[0]['discharge_date']).date()
        
        if 'gender' in df.columns:
            patient_data['gender'] = df.iloc[0]['gender']
        
        # Extract survey responses
        survey_responses = {}
        for col in df.columns:
            if col.startswith('response_'):
                question = col.replace('response_', '')
                survey_responses[question] = df.iloc[0][col]
        
        patient_data['survey_responses'] = survey_responses
        
        return patient_data
    except Exception as e:
        st.error(f"Error loading patient data: {str(e)}")
        return None

def generate_pdf_report():
    """Generate a PDF report of the patient's survey and rating results using ReportLab"""
    try:
        # Import ReportLab for PDF generation
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.lib.units import inch
    except ImportError:
        # Fall back to text-only report if ReportLab is not available
        return generate_text_report()
    
    patient_data = st.session_state.patient_data
    predictions = patient_data.get('predictions', {})
    
    if not predictions:
        st.error("No rating data available to generate report.")
        return None, None
    
    # Create a buffer for the PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.alignment = 1  # Center alignment
    
    subtitle_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    # Create custom styles
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.darkblue,
        spaceAfter=12
    )
    
    info_style = ParagraphStyle(
        "InfoStyle",
        parent=styles["Normal"],
        fontSize=11,
        leftIndent=5
    )
    
    # Container for elements to add to the PDF
    elements = []
    
    # Add title
    elements.append(Paragraph("Hospital Patient Experience Report", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add patient information table
    patient_info = [
        ["Patient Information", ""],
        ["Patient ID:", patient_data.get('patient_id', 'Not provided')],
        ["Facility Name:", patient_data.get('facility_name', 'Not provided')],
        ["Facility ID:", patient_data.get('facility_id', 'Not provided')],
    ]
    
    if patient_data.get('admission_date'):
        patient_info.append(["Admission Date:", patient_data['admission_date'].strftime('%Y-%m-%d')])
    
    if patient_data.get('discharge_date'):
        patient_info.append(["Discharge Date:", patient_data['discharge_date'].strftime('%Y-%m-%d')])
    
    if patient_data.get('gender'):
        patient_info.append(["Gender:", patient_data['gender']])
        
    patient_info.append(["Report Generated:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
    # Create patient info table
    info_table = Table(patient_info, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONT', (0, 0), (1, 0), 'Helvetica-Bold', 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add overall rating information
    elements.append(Paragraph("Overall Hospital Rating", header_style))
    
    star_rating = predictions.get('star_rating', 0)
    stars = "★" * star_rating + "☆" * (5 - star_rating)
    
    rating_text = f"This hospital received {star_rating} out of 5 stars based on patient experience survey data."
    elements.append(Paragraph(f"<b>{stars}</b> - {rating_text}", normal_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add category scores
    responses = patient_data.get('survey_responses', {})
    if responses:
        elements.append(Paragraph("Category Performance", header_style))
        
        # Calculate scores
        nurse_score = calculate_section_scores(responses, 'nurse_communication')
        doctor_score = calculate_section_scores(responses, 'doctor_communication')
        hospital_env_score = calculate_section_scores(responses, 'hospital_environment')
        experience_score = calculate_section_scores(responses, 'hospital_experience')
        discharge_score = calculate_section_scores(responses, 'discharge')
        care_transition_score = calculate_section_scores(responses, 'care_transition')
        
        # Create score data for table
        score_data = [
            ["Category", "Score", "Performance Level"],
            ["Nurse Communication", f"{nurse_score:.1f}%", get_performance_level(nurse_score)],
            ["Doctor Communication", f"{doctor_score:.1f}%", get_performance_level(doctor_score)],
            ["Hospital Environment", f"{hospital_env_score:.1f}%", get_performance_level(hospital_env_score)],
            ["Hospital Experience", f"{experience_score:.1f}%", get_performance_level(experience_score)],
            ["Discharge Information", f"{discharge_score:.1f}%", get_performance_level(discharge_score)],
            ["Care Transition", f"{care_transition_score:.1f}%", get_performance_level(care_transition_score)]
        ]
        
        # Create score table
        score_table = Table(score_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 12),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),  # Center-align the score column
            ('ALIGN', (2, 1), (2, -1), 'CENTER'),  # Center-align the performance level column
        ]))
        
        # Add color coding based on performance level
        for i in range(1, len(score_data)):
            level = score_data[i][2]
            if level == "Excellent":
                score_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), colors.lightgreen)]))
            elif level == "Good":
                score_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), colors.lightblue)]))
            elif level == "Average":
                score_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), colors.yellow)]))
            else:  # Needs Improvement
                score_table.setStyle(TableStyle([('BACKGROUND', (2, i), (2, i), colors.salmon)]))
        
        elements.append(score_table)
        elements.append(Spacer(1, 0.25*inch))
        
    # Add recommendations section
    elements.append(Paragraph("Improvement Recommendations", header_style))
    
    # Get recommendations
    recs = get_improvement_recommendations(responses)
    
    if not recs:
        elements.append(Paragraph("No specific recommendations available at this time.", normal_style))
    else:
        for category, rec_list in recs.items():
            elements.append(Paragraph(f"<b>{category}</b>", subtitle_style))
            for rec in rec_list:
                elements.append(Paragraph(f"• {rec}", normal_style))
            elements.append(Spacer(1, 0.1*inch))
    
    # Add footer with disclaimer
    elements.append(Spacer(1, 0.5*inch))
    disclaimer = "Disclaimer: This report is based on survey data collected during the patient's hospital stay. It is intended for quality improvement purposes and should be reviewed by hospital staff."
    elements.append(Paragraph(disclaimer, info_style))
    
    # Build the PDF document
    doc.build(elements)
    
    # Prepare the PDF for download
    buffer.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hospital_report_{patient_data['patient_id']}_{timestamp}.pdf"
    
    return buffer, filename

def get_performance_level(score):
    """Return a performance level description based on a percentage score"""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Average"
    else:
        return "Needs Improvement"
        
def generate_text_report():
    """Generate a simple text report if ReportLab is not available"""
    patient_data = st.session_state.patient_data
    predictions = patient_data.get('predictions', {})
    
    if not predictions:
        st.error("No rating data available to generate report.")
        return None, None
    
    # Create a string with report content
    buffer = io.BytesIO()
    
    report_text = f"""
    HOSPITAL PATIENT EXPERIENCE REPORT
    
    Patient ID: {patient_data.get('patient_id', 'Not provided')}
    Facility: {patient_data.get('facility_name', 'Not provided')}
    Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    RATING SUMMARY
    Overall Rating: {predictions.get('star_rating', 0)} out of 5 Stars
    
    CATEGORY PERFORMANCE
    """
    
    # Add section scores if available
    responses = patient_data.get('survey_responses', {})
    if responses:
        nurse_score = calculate_section_scores(responses, 'nurse_communication')
        doctor_score = calculate_section_scores(responses, 'doctor_communication')
        hospital_env_score = calculate_section_scores(responses, 'hospital_environment')
        experience_score = calculate_section_scores(responses, 'hospital_experience')
        discharge_score = calculate_section_scores(responses, 'discharge')
        care_transition_score = calculate_section_scores(responses, 'care_transition')
        
        report_text += f"""
        Nurse Communication: {nurse_score:.1f}% - {get_performance_level(nurse_score)}
        Doctor Communication: {doctor_score:.1f}% - {get_performance_level(doctor_score)}
        Hospital Environment: {hospital_env_score:.1f}% - {get_performance_level(hospital_env_score)}
        Hospital Experience: {experience_score:.1f}% - {get_performance_level(experience_score)}
        Discharge Information: {discharge_score:.1f}% - {get_performance_level(discharge_score)}
        Care Transition: {care_transition_score:.1f}% - {get_performance_level(care_transition_score)}
        """
    
    # Add recommendations
    report_text += "\nIMPROVEMENT RECOMMENDATIONS\n"
    recs = get_improvement_recommendations(responses)
    
    if not recs:
        report_text += "No specific recommendations available at this time.\n"
    else:
        for category, rec_list in recs.items():
            report_text += f"\n{category}:\n"
            for rec in rec_list:
                report_text += f"- {rec}\n"
    
    buffer.write(report_text.encode())
    buffer.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hospital_report_{patient_data['patient_id']}_{timestamp}.txt"
    
    return buffer, filename

def generate_historical_trend_chart(hospital_data):
    """Generate a trend chart showing hospital rating over time"""
    # For demo purposes - in a real app, you would load historical data from a database
    # We'll simulate data here
    
    dates = pd.date_range(end=datetime.now(), periods=12, freq='M').tolist()
    ratings = [3, 3, 4, 3, 4, 4, 5, 4, 4, 3, 4, 5]
    
    # Create a DataFrame with the dates and ratings
    df = pd.DataFrame({
        'date': dates,
        'rating': ratings
    })
    
    # Create a line chart with Plotly
    fig = px.line(
        df, 
        x='date', 
        y='rating',
        title="Historical Rating Trend",
        labels={"date": "Month", "rating": "Star Rating"},
        markers=True
    )
    
    # Customize the chart
    fig.update_layout(
        xaxis=dict(tickformat="%b %Y"),
        yaxis=dict(range=[0, 5.5], dtick=1),
        hovermode="x unified"
    )
    
    # Add the current rating if available
    if 'predictions' in st.session_state.patient_data and 'star_rating' in st.session_state.patient_data['predictions']:
        current_rating = st.session_state.patient_data['predictions']['star_rating']
        
        fig.add_trace(
            go.Scatter(
                x=[datetime.now()],
                y=[current_rating],
                mode='markers',
                marker=dict(color='red', size=12),
                name='Current Rating'
            )
        )
    
    return fig

def generate_comparative_bar_chart():
    """Generate a comparative bar chart showing hospital performance against regional/national averages"""
    # For demo purposes - in a real app, you would load benchmark data from a database
    
    # Calculate current scores from survey responses
    responses = st.session_state.patient_data.get('survey_responses', {})
    
    if not responses:
        return None
    
    # Define the categories to plot
    categories = [
        'Nurse Communication',
        'Doctor Communication',
        'Hospital Environment',
        'Hospital Experience',
        'Discharge Information',
        'Care Transition'
    ]
    
    # Calculate current scores
    current_scores = [
        calculate_section_scores(responses, 'nurse_communication'),
        calculate_section_scores(responses, 'doctor_communication'),
        calculate_section_scores(responses, 'hospital_environment'),
        calculate_section_scores(responses, 'hospital_experience'),
        calculate_section_scores(responses, 'discharge'),
        calculate_section_scores(responses, 'care_transition')
    ]
    
    # Generate simulated regional and national averages
    # In a real app, this would come from a database
    regional_avg = [85, 80, 75, 78, 82, 74]
    national_avg = [82, 78, 72, 75, 80, 70]
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'Category': categories + categories + categories,
        'Score': current_scores + regional_avg + national_avg,
        'Type': ['Current Hospital']*6 + ['Regional Average']*6 + ['National Average']*6
    })
    
    # Create a grouped bar chart
    fig = px.bar(
        df,
        x='Category',
        y='Score',
        color='Type',
        barmode='group',
        title='Hospital Performance Comparison',
        color_discrete_map={
            'Current Hospital': '#4682B4',
            'Regional Average': '#20B2AA',
            'National Average': '#778899'
        },
        text_auto=True
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title=None,
        yaxis_title='Score (%)',
        yaxis=dict(range=[0, 105]),
        legend_title=None,
    )
    
    fig.update_traces(textposition='outside')
    
    return fig

def main():
    # Page header
    st.markdown('<div class="main-header">Hospital Patient Experience Rating System</div>', unsafe_allow_html=True)
    
    # Load the model
    model, scaler, selected_features = load_model()
    
    if model is None or scaler is None or selected_features is None:
        st.error("Failed to load the prediction model. Please check the model files.")
        return
    
    # Add a sidebar for data management and model selection
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_type = st.sidebar.radio(
        "Select Rating Model",
        options=["Standard Model", "Enhanced Model"],
        index=1,
        help="Choose which model to use for patient rating prediction"
    )
    
    # Update session state with model type
    st.session_state.model_type = 'enhanced' if model_type == "Enhanced Model" else 'standard'
    
    st.sidebar.header("Data Management")
    
    # Option to upload patient data
    uploaded_file = st.sidebar.file_uploader("Upload Patient Data", type=["csv"])
    
    if uploaded_file is not None:
        if st.sidebar.button("Load Patient Data"):
            patient_data = load_patient_data_from_csv(uploaded_file)
            if patient_data:
                st.session_state.patient_data = patient_data
                st.sidebar.success("Patient data loaded successfully!")
                st.experimental_rerun()
    
    # Option to save data
    if st.session_state.patient_data.get('patient_id') and st.session_state.patient_data.get('survey_responses'):
        st.sidebar.subheader("Save Current Patient Data")
        if st.sidebar.button("Generate Download Link"):
            href = save_patient_data_to_csv()
            if href:
                st.sidebar.markdown(href, unsafe_allow_html=True)
    
    # Option to reset all data
    if st.sidebar.button("Reset All Patient Data"):
        st.session_state.patient_data = {
            'patient_id': '',
            'facility_id': '',
            'facility_name': '',
            'admission_date': None,
            'discharge_date': None,
            'survey_responses': {},
            'predictions': {}
        }
        st.sidebar.success("Patient data reset successfully!")
        st.experimental_rerun()
    
    # Create tabs for the different sections
    tab1, tab2, tab3 = st.tabs(["Patient Report", "Patient Survey", "HCAHPS Form"])
    
    with tab1:
        st.markdown('<div class="sub-header">Patient Report & Analytics</div>', unsafe_allow_html=True)
        
        if not st.session_state.patient_data['survey_responses']:
            st.info("No survey data available. Please complete the HCAHPS survey in the 'Patient Survey' tab.")
        else:
            # Display patient information
            st.markdown("### Patient Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Patient ID:** {st.session_state.patient_data.get('patient_id', 'Not provided')}")
                st.write(f"**Facility:** {st.session_state.patient_data.get('facility_name', 'Not provided')}")
            
            with col2:
                admission_date = st.session_state.patient_data.get('admission_date')
                discharge_date = st.session_state.patient_data.get('discharge_date')
                
                if admission_date:
                    st.write(f"**Admission Date:** {admission_date.strftime('%b %d, %Y')}")
                else:
                    st.write("**Admission Date:** Not provided")
                
                if discharge_date:
                    st.write(f"**Discharge Date:** {discharge_date.strftime('%b %d, %Y')}")
                else:
                    st.write("**Discharge Date:** Not provided")
            
            # Get prediction based on survey responses
            responses = st.session_state.patient_data['survey_responses']
            prediction_result = st.session_state.patient_data.get('predictions', {})
            
            if not prediction_result:
                try:
                    star_rating, prediction_proba, features = predict_rating(responses, model, scaler, selected_features)
                    prediction_result = {
                        'star_rating': star_rating,
                        'prediction_proba': prediction_proba,
                        'features': features,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.patient_data['predictions'] = prediction_result
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    return
            
            # Display the prediction result
            star_rating = prediction_result.get('star_rating', 0)
            features = prediction_result.get('features', {})
            
            st.markdown('<div class="rating-card">', unsafe_allow_html=True)
            st.subheader("Predicted Patient Experience Rating")
            
            # Visual star representation
            stars_html = "⭐" * int(star_rating) + "☆" * (5 - int(star_rating))
            st.markdown(f"<h1 style='text-align: center;'>{stars_html}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{star_rating:.0f} out of 5 Stars</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Rating interpretation
            st.subheader("Rating Interpretation")
            
            rating_interpretations = {
                1: "The patient experience was significantly below average. Multiple areas need immediate improvement.",
                2: "The patient experience was below average. Several aspects of care need enhancement.",
                3: "The patient experienced average care. Some improvements could enhance future experiences.",
                4: "The patient experience was above average. Minor improvements could help achieve excellence.",
                5: "The patient experienced excellent care. Focus on maintaining these high standards."
            }
            
            st.info(rating_interpretations[int(star_rating)])
            
            # Section scores
            st.subheader("Category Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate section scores
                nurse_score = calculate_section_scores(responses, 'nurse_communication')
                doctor_score = calculate_section_scores(responses, 'doctor_communication')
                hospital_env_score = calculate_section_scores(responses, 'hospital_environment')
                experience_score = calculate_section_scores(responses, 'hospital_experience')
                
                display_section_score("Nurse Communication", nurse_score)
                display_section_score("Doctor Communication", doctor_score)
                display_section_score("Hospital Environment", hospital_env_score)
                display_section_score("Hospital Experience", experience_score)
            
            with col2:
                discharge_score = calculate_section_scores(responses, 'discharge')
                care_transition_score = calculate_section_scores(responses, 'care_transition')
                recommend_score = 100 if responses.get('recommend_hospital') == 'Definitely yes' else \
                                  75 if responses.get('recommend_hospital') == 'Probably yes' else \
                                  25 if responses.get('recommend_hospital') == 'Probably no' else \
                                  0 if responses.get('recommend_hospital') == 'Definitely no' else 0
                
                display_section_score("Discharge Information", discharge_score)
                display_section_score("Care Transition", care_transition_score)
                display_section_score("Would Recommend", recommend_score)
            
            # Performance visualization
            st.subheader("Performance Visualization")
            
            tabs = st.tabs(["Current Performance", "Historical Trend", "Benchmark Comparison"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Generate and display radar chart
                    radar_fig = generate_radar_chart(features)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with col2:
                    # Generate categories for bar chart
                    categories = ['Nurse\nCommunication', 'Doctor\nCommunication', 'Hospital\nEnvironment', 
                                'Hospital\nExperience', 'Discharge\nInfo', 'Care\nTransition']
                    scores = [nurse_score, doctor_score, hospital_env_score, experience_score, discharge_score, care_transition_score]
                    
                    fig = px.bar(
                        x=categories, 
                        y=scores,
                        title="Category Performance Scores",
                        labels={"x": "", "y": "Score (%)"},
                        color=scores,
                        color_continuous_scale="blues",
                        text=[f"{score:.1f}%" for score in scores]
                    )
                    
                    fig.update_traces(textposition='outside')
                    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    
                    # Add a target line at 80%
                    fig.add_shape(type="line",
                        x0=-0.5, y0=80, x1=len(categories)-0.5, y1=80,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    fig.add_annotation(
                        x=len(categories)-1, y=82,
                        text="Target (80%)",
                        showarrow=False,
                        font=dict(color="red")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            with tabs[1]:
                # Historical trend chart
                st.info("This chart shows the hospital's rating trend over the past 12 months.")
                trend_fig = generate_historical_trend_chart(st.session_state.patient_data)
                st.plotly_chart(trend_fig, use_container_width=True)
                
                # Add note about data source
                st.caption("Note: Historical data shown here is for demonstration purposes only.")
            
            with tabs[2]:
                # Comparative bar chart
                st.info("This chart compares the hospital's performance against regional and national benchmarks.")
                comp_fig = generate_comparative_bar_chart()
                if comp_fig:
                    st.plotly_chart(comp_fig, use_container_width=True)
                    
                    # Add note about data source
                    st.caption("Note: Benchmark data shown here is simulated for demonstration purposes only.")
            
            # Improvement recommendations
            recommendations = get_improvement_recommendations(features)
            
            if recommendations:
                st.subheader("Areas for Improvement")
                
                for i, rec in enumerate(recommendations):
                    with st.expander(f"{rec['category']} (Current: {rec['score']})"):
                        st.markdown("### Recommendations:")
                        for tip in rec['tips']:
                            st.markdown(f"- {tip}")
            
            # Generate a report and provide a download button
            report_buffer, report_filename = generate_pdf_report()
            if report_buffer:
                st.download_button(
                    label="Download Patient Experience Report",
                    data=report_buffer,
                    file_name=report_filename,
                    mime="text/plain",
                    help="Download a report with the complete analysis of patient experience"
                )
            
            # Historical trend chart
            st.subheader("Historical Rating Trend")
            
            trend_fig = generate_historical_trend_chart(st.session_state.patient_data)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Comparative bar chart
            st.subheader("Performance Comparison")
            
            comp_fig = generate_comparative_bar_chart()
            if comp_fig:
                st.plotly_chart(comp_fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Patient Survey</div>', unsafe_allow_html=True)
        st.markdown("Please complete the following patient information and survey questions.")
        
        # Patient info form
        patient_info = patient_info_form()
        
        # Check if we have enough survey responses (at least 5)
        enough_responses = len(st.session_state.patient_data.get('survey_responses', {})) >= 5
        
        # Display a progress indicator for the survey completion
        if st.session_state.patient_data.get('survey_responses'):
            num_responses = len(st.session_state.patient_data['survey_responses'])
            st.progress(min(1.0, num_responses / 10), text=f"Survey completion: {num_responses}/10 questions")
            
            if num_responses < 5:
                st.info("Please answer at least 5 questions to generate a rating")
        
        # Survey form
        st.markdown('<div class="section-title">HCAHPS Survey Questions</div>', unsafe_allow_html=True)
        st.markdown("Please answer the following key questions about your hospital stay.")
        
        # Create a simple version of the survey with key questions
        with st.expander("Quick Survey Questions", expanded=True):
            # Nurse Communication
            st.subheader("Nurse Communication")
            nurse_q = st.radio(
                "During this hospital stay, how often did nurses treat you with courtesy and respect?",
                options=["", "Never", "Sometimes", "Usually", "Always"],
                index=0,
                key="nurse_respect",
                horizontal=True
            )
            
            if nurse_q:
                st.session_state.patient_data['survey_responses']["nurse_respect"] = nurse_q
            
            # Doctor Communication
            st.subheader("Doctor Communication")
            doctor_q = st.radio(
                "During this hospital stay, how often did doctors explain things in a way you could understand?",
                options=["", "Never", "Sometimes", "Usually", "Always"],
                index=0,
                key="doctor_explain",
                horizontal=True
            )
            
            if doctor_q:
                st.session_state.patient_data['survey_responses']["doctor_explain"] = doctor_q
                
            # Hospital Environment
            st.subheader("Hospital Environment")
            env_q = st.radio(
                "During this hospital stay, how often was your room and bathroom kept clean?",
                options=["", "Never", "Sometimes", "Usually", "Always"],
                index=0,
                key="room_clean",
                horizontal=True
            )
            
            if env_q:
                st.session_state.patient_data['survey_responses']["room_clean"] = env_q
                
            # Hospital Experience
            st.subheader("Hospital Experience")
            exp_q = st.radio(
                "How often was the area around your room quiet at night?",
                options=["", "Never", "Sometimes", "Usually", "Always"],
                index=0,
                key="quiet_night",
                horizontal=True
            )
            
            if exp_q:
                st.session_state.patient_data['survey_responses']["quiet_night"] = exp_q
            
            # Care Transition
            st.subheader("Care Transition")
            transition_q = st.radio(
                "During this hospital stay, staff took my preferences and those of my family/caregiver into account in deciding what my health care needs would be when I left.",
                options=["", "Strongly Disagree", "Disagree", "Agree", "Strongly Agree"],
                index=0,
                key="preferences_considered",
                horizontal=True
            )
            
            if transition_q:
                st.session_state.patient_data['survey_responses']["preferences_considered"] = transition_q
                
        st.markdown("For a complete survey, please use the **HCAHPS Form** tab.")
        
        # Submit button for survey
        submit_disabled = not (patient_info.get('is_valid', False) and enough_responses)
        
        submit_button = st.button(
            "Submit Survey and Calculate Rating",
            type="primary", 
            disabled=submit_disabled,
            help="Complete required fields to enable" if submit_disabled else "Submit survey to calculate hospital rating"
        )
        
        if submit_disabled:
            if not patient_info.get('is_valid', False):
                st.warning("Please complete all required patient information fields.")
            if not enough_responses:
                st.warning("Please answer at least 5 survey questions.")
                
        if submit_button:
            # Process the survey
            st.success("Survey submitted successfully! View your results in the Patient Report tab.")
            
            # Calculate rating
            try:
                responses = st.session_state.patient_data['survey_responses']
                star_rating, prediction_proba, features = predict_rating(responses, model, scaler, selected_features)
                
                # Save prediction to session state
                st.session_state.patient_data['predictions'] = {
                    'star_rating': star_rating,
                    'prediction_proba': prediction_proba,
                    'features': features,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Show a success message with the rating
                st.balloons()
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>Hospital Rating: {'⭐' * int(star_rating)}</h2>
                    <p>The predicted rating is {star_rating} out of 5 stars.</p>
                    <p>Please visit the <b>Patient Report</b> tab to see the detailed analysis.</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error calculating rating: {str(e)}")
    
    with tab3:
        st.markdown('<div class="sub-header">HCAHPS Survey Form</div>', unsafe_allow_html=True)
        st.markdown("""
        This tab contains the complete HCAHPS (Hospital Consumer Assessment of Healthcare Providers and Systems) survey form. 
        Please fill out all questions to provide feedback on your hospital experience.
        """)
        
        # Display full HCAHPS survey form
        responses = hcahps_survey_form()
        
        # Submit button
        if st.button("Submit HCAHPS Form", type="primary"):
            # Check if we have responses
            if not responses:
                st.error("Please answer at least some of the survey questions.")
            else:
                st.success("HCAHPS survey submitted successfully! View your results in the Patient Report tab.")
                
                # Save responses to session state
                st.session_state.patient_data['survey_responses'] = responses
                
                # Calculate rating
                try:
                    star_rating, prediction_proba, features = predict_rating(responses, model, scaler, selected_features)
                    
                    # Save prediction to session state
                    st.session_state.patient_data['predictions'] = {
                        'star_rating': star_rating,
                        'prediction_proba': prediction_proba,
                        'features': features,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                except Exception as e:
                    st.error(f"Error calculating rating: {str(e)}")
