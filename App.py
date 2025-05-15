import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Hospital Rating Predictor",
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
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #4682B4;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .feature-importance {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .prediction-result {
        background-color: #f0fff0;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .improvement-tips {
        background-color: #fff0f0;
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and related files
@st.cache_resource
def load_model():
    model = joblib.load('hospital_rating_model_enhanced.pkl')
    scaler = joblib.load('scaler_enhanced.pkl')
    selected_features = joblib.load('selected_features.pkl')
    return model, scaler, selected_features

# Load sample data for comparison
@st.cache_data
def load_sample_data():
    """Load and preprocess sample hospital data for comparison benchmarks"""
    try:
        # Check if file exists first
        if os.path.exists('Hospital_Rating_Dataset.csv'):
            df = pd.read_csv('Hospital_Rating_Dataset.csv')
            
            # Just extract the rating data for benchmarking
            ratings = df[df['HCAHPS_measure_id'] == 'H_STAR_RATING'][['Patient_Survey_Star_Rating']]
            ratings = ratings.dropna()
            ratings = pd.to_numeric(ratings['Patient_Survey_Star_Rating'], errors='coerce')
            
            # Get distribution
            distribution = ratings.value_counts().sort_index()
            return distribution
        else:
            # Return a simulated distribution if file doesn't exist
            return pd.Series({1: 10, 2: 25, 3: 40, 4: 20, 5: 5})
    except Exception as e:
        st.warning(f"Could not load sample data: {e}")
        # Return a simulated distribution
        return pd.Series({1: 10, 2: 25, 3: 40, 4: 20, 5: 5})

def predict_rating_from_inputs(inputs, model, scaler, selected_features):
    """
    Process inputs and make a prediction with proper feature engineering based on Model2.py
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
                u_value = inputs.get('usually_pct', 25)  # Usually percentage
                # More accurate calculation: SN = 100 - Always - Usually
                df[feature] = max(0, min(100 - a_value - u_value, 30))  # Cap at reasonable values
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
    
    # Handle special calculated features - match Model2.py feature engineering
    if 'avg_always_percent' in selected_features:
        df['avg_always_percent'] = inputs.get('avg_always_percent', 75)
    
    if 'max_always_percent' in selected_features:
        # Usually about 10-15% higher than the average
        df['max_always_percent'] = min(100, inputs.get('avg_always_percent', 75) * 1.15)
    
    if 'avg_sn_percent' in selected_features:
        df['avg_sn_percent'] = inputs.get('avg_sn_percent', 15)
    
    if 'nurse_doctor_interaction' in selected_features:
        # From Model2.py: df['nurse_doctor_interaction'] = df['avg_nurse_percent'] * df['avg_doctor_percent']
        df['nurse_doctor_interaction'] = inputs.get('avg_nurse_percent', 80) * inputs.get('avg_doctor_percent', 75)
    
    if 'always_to_sn_ratio' in selected_features:
        # From Model2.py: df['always_to_sn_ratio'] = df['avg_always_percent'] / (df['avg_sn_percent'] + 1e-5)
        df['always_to_sn_ratio'] = inputs.get('avg_always_percent', 75) / (inputs.get('avg_sn_percent', 15) + 1e-5)
    
    if 'weighted_rating' in selected_features:
        # This is critical for prediction: it's the product of estimated rating and response rate
        # Estimate the rating first based on key metrics
        nurse_pct = inputs.get('avg_nurse_percent', 80)
        doctor_pct = inputs.get('avg_doctor_percent', 75)
        sn_pct = inputs.get('avg_sn_percent', 15)
        
        # The response rate is important for this weight
        response_rate = inputs.get('response_rate', 30)
        
        # A more accurate way to estimate the star rating (0-4 scale for raw prediction)
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
    
    if 'normalized_surveys' in selected_features:
        # From Model2.py: df['normalized_surveys'] = df['Number_of_Completed_Surveys'] / df['Number_of_Completed_Surveys'].max()
        # We'll assume max surveys is 1000 for normalization
        df['normalized_surveys'] = inputs.get('Number_of_Completed_Surveys', 300) / 1000
    
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

def generate_radar_chart(inputs):
    """Generate a radar chart for hospital metrics visualization"""
    categories = [
        'Nurse Communication', 'Doctor Communication', 
        'Staff Responsiveness', 'Medication Communication',
        'Cleanliness', 'Pain Management'
    ]
    
    values = [
        inputs.get('avg_nurse_percent', 80),
        inputs.get('avg_doctor_percent', 75),
        inputs.get('responsiveness_pct', 65),
        inputs.get('medication_pct', 70),
        inputs.get('cleanliness', 70),
        inputs.get('pain_mgmt', 75)
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
        name='Hospital Metrics'
    ))
    
    # Add benchmark line (90% is typically excellent)
    fig.add_trace(go.Scatterpolar(
        r=[90, 90, 90, 90, 90, 90, 90],
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

def create_rating_distribution_chart(distribution, predicted_rating):
    """Create a bar chart of hospital rating distribution with the predicted rating highlighted"""
    # Create a bar chart
    fig = px.bar(
        x=distribution.index, 
        y=distribution.values,
        labels={'x': 'Star Rating', 'y': 'Number of Hospitals'},
        title='Hospital Rating Distribution'
    )
    
    # Highlight the predicted rating
    fig.add_shape(
        type="rect",
        x0=predicted_rating-0.4, y0=0,
        x1=predicted_rating+0.4, y1=distribution.max(),
        line=dict(color="rgba(0,0,0,0)"),
        fillcolor="rgba(255,255,0,0.3)"
    )
    
    # Add a marker for "Your Hospital"
    fig.add_annotation(
        x=predicted_rating,
        y=distribution[int(predicted_rating)] if int(predicted_rating) in distribution.index else 0,
        text="Your Hospital",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-40
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            title_font=dict(size=16),
        ),
        yaxis=dict(
            title_font=dict(size=16),
        ),
        title_font=dict(size=18),
    )
    
    return fig

def get_category_recommendations(inputs, star_rating):
    """Generate specific recommendations for each category based on inputs and predicted rating"""
    recommendations = []
    
    # Nurse communication
    nurse_pct = inputs.get('avg_nurse_percent', 80)
    if nurse_pct < 85:
        recommendations.append({
            'category': 'Nurse Communication',
            'current': nurse_pct,
            'target': 90,
            'tips': [
                "Implement hourly nurse rounding protocols",
                "Provide communication skills training for nursing staff",
                "Establish bedside shift reporting to involve patients"
            ]
        })
    
    # Doctor communication
    doctor_pct = inputs.get('avg_doctor_percent', 75)
    if doctor_pct < 85:
        recommendations.append({
            'category': 'Doctor Communication',
            'current': doctor_pct,
            'target': 90,
            'tips': [
                "Implement AIDET (Acknowledge, Introduce, Duration, Explanation, Thank you) protocol",
                "Schedule dedicated time for doctor-patient discussions",
                "Provide communication skills workshops for physicians"
            ]
        })
    
    # Responsiveness
    resp_pct = inputs.get('responsiveness_pct', 65)
    if resp_pct < 80:
        recommendations.append({
            'category': 'Staff Responsiveness',
            'current': resp_pct,
            'target': 85,
            'tips': [
                "Implement call light rapid response teams",
                "Set standards for maximum response times",
                "Use technology to improve response time tracking"
            ]
        })
    
    # Medication communication
    med_pct = inputs.get('medication_pct', 70)
    if med_pct < 80:
        recommendations.append({
            'category': 'Medication Communication',
            'current': med_pct,
            'target': 85,
            'tips': [
                "Use teach-back method when explaining medications",
                "Provide written medication information in patient-friendly language",
                "Schedule medication education at optimal times for patient comprehension"
            ]
        })
    
    # Cleanliness
    clean_pct = inputs.get('cleanliness', 70)
    if clean_pct < 80:
        recommendations.append({
            'category': 'Hospital Cleanliness',
            'current': clean_pct,
            'target': 85,
            'tips': [
                "Implement visual cleanliness inspections with feedback",
                "Ensure regular cleaning of high-touch surfaces",
                "Train staff on proper cleaning protocols"
            ]
        })
    
    return recommendations

def main():
    st.markdown('<div class="main-header">Hospital Rating Predictor</div>', unsafe_allow_html=True)
    st.markdown("""
    This application uses machine learning to predict a hospital's star rating (1-5 stars) based on patient survey data.
    The model has been trained on HCAHPS (Hospital Consumer Assessment of Healthcare Providers and Systems) data.
    """)
    
    # Load the model
    model, scaler, selected_features = load_model()
    
    # Load sample distribution data
    distribution = load_sample_data()
    
    # Create a sidebar for inputs
    st.sidebar.markdown('<div class="sub-header">Hospital Survey Data Input</div>', unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Basic Metrics", "Advanced Metrics", "Hospital Environment"])
    
    with tab1:
        st.markdown('<div class="sub-header">Basic Survey Metrics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Always responses (most important feature from our model)
            avg_always_percent = st.slider(
                "Average 'Always' Response Percentage", 
                min_value=0.0, 
                max_value=100.0, 
                value=75.0,
                help="Average percentage of patients who answered 'Always' to questions about hospital quality"
            )
            
            # Usually responses (new addition for more accuracy)
            usually_pct = st.slider(
                "Average 'Usually' Response Percentage", 
                min_value=0.0, 
                max_value=100.0, 
                value=20.0,
                help="Average percentage of patients who answered 'Usually' to questions about hospital quality"
            )
            
            # Sometimes/Never responses (another top feature)
            avg_sn_percent = st.slider(
                "Average 'Sometimes/Never' Response Percentage", 
                min_value=0.0, 
                max_value=100.0, 
                value=100.0 - avg_always_percent - usually_pct,
                help="Average percentage of patients who answered 'Sometimes' or 'Never' to questions about hospital quality"
            )
        
        with col2:
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
            
            # Hospital type
            hospital_type = st.selectbox(
                "Hospital Type",
                options=["Community Hospital", "Academic Medical Center", "Specialty Hospital", "Critical Access Hospital"],
                index=0,
                help="Type of hospital (for context)"
            )
    
    with tab2:
        st.markdown('<div class="sub-header">Category-Specific Metrics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
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
            
            # Medication explanation
            medication_pct = st.slider(
                "Medication Explanation Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=65.0,
                help="Percentage of positive responses regarding medication explanation"
            )
        
        with col2:
            # Hospital cleanliness
            cleanliness = st.slider(
                "Hospital Cleanliness Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=70.0,
                help="Percentage of positive responses regarding hospital cleanliness"
            )
            
            # Responsiveness of staff
            responsiveness_pct = st.slider(
                "Staff Responsiveness Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=65.0,
                help="Percentage of positive responses regarding staff responsiveness"
            )
            
            # Pain management (new metric)
            pain_mgmt = st.slider(
                "Pain Management Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=75.0,
                help="Percentage of positive responses regarding pain management"
            )
    
    with tab3:
        st.markdown('<div class="sub-header">Hospital Environment</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quietness
            quietness = st.slider(
                "Hospital Quietness Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=60.0,
                help="Percentage of positive responses regarding quietness of hospital environment"
            )
            
            # Discharge information
            discharge_info = st.slider(
                "Discharge Information Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=85.0,
                help="Percentage of positive responses regarding discharge information provided"
            )
        
        with col2:
            # Care transition
            care_transition = st.slider(
                "Care Transition Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=70.0,
                help="Percentage of positive responses regarding care transition"
            )
            
            # Overall hospital rating (direct)
            overall_rating = st.slider(
                "Direct Overall Hospital Rating (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=75.0,
                help="Percentage of patients giving a high overall rating to the hospital"
            )
    
    # Adding a timestamp to track when the prediction was made
    prediction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Button to make prediction
    if st.button("Predict Hospital Rating", type="primary"):
        # Collect all inputs into a single dictionary
        inputs = {
            'avg_always_percent': avg_always_percent,
            'avg_sn_percent': avg_sn_percent,
            'usually_pct': usually_pct,
            'Number_of_Completed_Surveys': num_surveys,
            'Survey_Response_Rate_Percent': response_rate,
            'response_rate': response_rate,
            'avg_nurse_percent': nurse_percent,
            'avg_doctor_percent': doctor_percent,
            'cleanliness': cleanliness,
            'medication_pct': medication_pct,
            'responsiveness_pct': responsiveness_pct,
            'pain_mgmt': pain_mgmt,
            'quietness': quietness,
            'discharge_info': discharge_info,
            'care_transition': care_transition,
            'overall_rating': overall_rating,
            'hospital_type': hospital_type
        }
        
        # Get prediction
        star_rating, prediction_proba, features_used = predict_rating_from_inputs(
            inputs, model, scaler, selected_features
        )
        
        # Display prediction in a nice format
        st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"<h1 style='text-align: center;'>Predicted Rating: {star_rating:.0f} ⭐</h1>", unsafe_allow_html=True)
            
            # Visual star representation
            stars_html = "⭐" * int(star_rating) + "☆" * (5 - int(star_rating))
            st.markdown(f"<h2 style='text-align: center;'>{stars_html}</h2>", unsafe_allow_html=True)
            
            # Confidence indicator if available
            if prediction_proba is not None:
                confidence = prediction_proba[int(star_rating)-1] * 100  # -1 because rating is 1-indexed
                st.markdown(f"<p style='text-align: center;'>Prediction confidence: {confidence:.1f}%</p>", unsafe_allow_html=True)
            
            st.markdown(f"<p style='text-align: center; color: #888;'>Prediction timestamp: {prediction_timestamp}</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display prediction interpretation
        st.subheader("Rating Interpretation")
        
        rating_interpretations = {
            1: "This hospital is significantly below average in patient satisfaction. Immediate improvements are recommended across multiple areas.",
            2: "This hospital is below average in patient satisfaction. Several areas need improvement to enhance patient experience.",
            3: "This hospital has average patient satisfaction. Some areas could be improved to enhance the overall experience.",
            4: "This hospital has above-average patient satisfaction. Minor improvements could help achieve excellent status.",
            5: "This hospital has excellent patient satisfaction. Focus should be on maintaining these high standards."
        }
        
        st.info(rating_interpretations[int(star_rating)])
        
        # Create two columns for the visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate and display radar chart
            radar_fig = generate_radar_chart(inputs)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            # Generate and display rating distribution
            dist_fig = create_rating_distribution_chart(distribution, star_rating)
            st.plotly_chart(dist_fig, use_container_width=True)
        
        # Get detailed recommendations
        recommendations = get_category_recommendations(inputs, star_rating)
        
        if recommendations:
            st.subheader("Improvement Recommendations")
            
            for i, rec in enumerate(recommendations):
                with st.expander(f"{i+1}. {rec['category']} (Current: {rec['current']:.1f}%, Target: {rec['target']}%)"):
                    for tip in rec['tips']:
                        st.markdown(f"- {tip}")
        
        # Show the most important feature values used
        with st.expander("Feature Importance Analysis"):
            # First identify the most important features
            top_features = [
                {'name': 'Average "Always" Responses', 'value': avg_always_percent, 'impact': 'High'},
                {'name': 'Average "Sometimes/Never" Responses', 'value': avg_sn_percent, 'impact': 'High'},
                {'name': 'Nurse Communication', 'value': nurse_percent, 'impact': 'High'},
                {'name': 'Doctor Communication', 'value': doctor_percent, 'impact': 'Medium'},
                {'name': 'Staff Responsiveness', 'value': responsiveness_pct, 'impact': 'Medium'},
                {'name': 'Hospital Cleanliness', 'value': cleanliness, 'impact': 'Medium'},
                {'name': 'Medication Communication', 'value': medication_pct, 'impact': 'Medium'},
                {'name': 'Survey Response Rate', 'value': response_rate, 'impact': 'Low'},
            ]
            
            # Create a formatted table
            st.markdown("### Most Important Factors Influencing the Prediction")
            
            # Format and display the table
            table_html = "<table width='100%' style='text-align: left;'>"
            table_html += "<tr><th>Feature</th><th>Current Value</th><th>Impact</th></tr>"
            
            for feature in top_features:
                impact_color = {
                    'High': '#ff6b6b',
                    'Medium': '#ffa94d',
                    'Low': '#69db7c'
                }[feature['impact']]
                
                table_html += f"<tr><td>{feature['name']}</td><td>{feature['value']:.1f}%</td>"
                table_html += f"<td style='color: {impact_color};'>{feature['impact']}</td></tr>"
            
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
        
        # Add option to download a PDF report (simulated for now)
        # st.download_button(
        #     label="Download Full Analysis Report",
        #     data="Hospital Rating Prediction Report would be generated here",
        #     file_name=f"hospital_rating_report_{prediction_timestamp.replace(' ', '_').replace(':', '-')}.pdf",
        #     mime="application/pdf",
        #     help="Download a PDF with the complete analysis of hospital ratings"
        # )

if __name__ == "__main__":
    main()
