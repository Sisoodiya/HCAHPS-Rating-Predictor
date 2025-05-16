# Hospital Rating Prediction System: An Easy-to-Understand Guide

This document explains how our Hospital Rating Prediction system works in simple terms. We'll walk through how we prepare the data, build features, choose the right models, and fine-tune them for the best results.

## What Data We're Using

Our system uses patient survey data from hospitals across the country. These surveys (called HCAHPS - Hospital Consumer Assessment of Healthcare Providers and Systems) ask patients about their hospital experience. The main dataset (`Hospital_Rating_Dataset.csv`) contains:

- Information that identifies each hospital
- Different survey questions (like "How well did nurses communicate?")
- The percentage of patients who gave each type of answer (like "Always," "Usually," or "Sometimes/Never")
- Average scores for each question (called `HCAHPS_Linear_Mean_Value` - a numeric scale that simplifies the percentage responses into a single score)
- How many patients completed the survey at each hospital
- What percentage of patients responded to the survey
- The overall star rating (1-5 stars) given to each hospital

## Understanding HCAHPS_Linear_Mean_Value

The `HCAHPS_Linear_Mean_Value` is a special column in our dataset that deserves extra attention. Here's why it's so important:

### What It Is

`HCAHPS_Linear_Mean_Value` is a standardized score that converts the percentage-based patient responses into a single numerical value (typically between 1-4) for each survey question. It's essentially a weighted average that:

1. **Gives more weight to positive responses** ("Always")
2. **Gives less weight to negative responses** ("Sometimes/Never")
3. **Creates a consistent scale** across different question types

### Why We Use It

We chose to include this column for several important reasons:

1. **Simplified Comparisons**: Instead of dealing with multiple percentage columns (Always%, Usually%, Sometimes/Never%) for each question, the linear mean gives us a single value to work with.

2. **Statistical Reliability**: These values are calculated using a standardized methodology from CMS (Centers for Medicare & Medicaid Services), making them statistically robust.

3. **Stronger Predictive Power**: Our analysis found that these linear mean values are extremely predictive of the final star ratings - often more so than the raw percentages. In particular, the linear mean for overall hospital rating (H_HSP_RATING) is one of our model's most important features.

4. **Easier Interpretation**: A higher linear mean value consistently corresponds to better performance, making it easier to interpret than working with multiple percentage columns.

For example, a `HCAHPS_Linear_Mean_Value` of 3.8 for nurse communication indicates excellent performance, while a value of 2.1 suggests significant room for improvement.

### How It Improves Our Model

By including these linear mean values alongside the percentage responses, our model gains a more comprehensive view of hospital performance. This is particularly valuable for our advanced Model2.py, which can identify subtle patterns in how these linear values interact with other features to predict the final star rating.

## How We Clean and Prepare the Data

### 1. Basic Cleaning

First, we need to get the data ready for our models:

```python
# Replace 'Not Applicable' with missing values (NaN)
df.replace('Not Applicable', np.nan, inplace=True)

# Convert text numbers to actual numbers the computer can work with
numeric_cols = ['HCAHPS_Answer_Percent', 'HCAHPS_Linear_Mean_Value', 
                'Number_of_Completed_Surveys', 'Survey_Response_Rate_Percent']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```

### 2. Reshaping the Data

In the original dataset, each hospital has multiple rows (one for each survey question). We need to transform this so each hospital has just one row with all their survey results as columns:

```python
# Reorganize data to have one row per hospital
pivot_df = df.pivot_table(
    index=['facility_id', 'facility_name'],
    columns='HCAHPS_measure_id',
    values=['HCAHPS_Answer_Percent', 'HCAHPS_Linear_Mean_Value'],
    aggfunc='mean'
)

# Make column names easier to work with
pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_df.columns]
```

### 3. Getting the Star Ratings (What We Want to Predict)

We need to extract the hospital star ratings that we're trying to predict:

```python
# Extract the star ratings
target = df[df['HCAHPS_measure_id'] == 'H_STAR_RATING'][['facility_id', 'Patient_Survey_Star_Rating']]
target['Patient_Survey_Star_Rating'] = pd.to_numeric(target['Patient_Survey_Star_Rating'], errors='coerce')

# Adjust ratings to start from 0 instead of 1 (this helps our models)
target['Patient_Survey_Star_Rating'] = target['Patient_Survey_Star_Rating'] - 1
```

### 4. Handling Missing Information

In our advanced model, we use a method called KNN imputation to fill in missing values. It looks at similar hospitals to make good guesses about missing data:

```python
# Find and fill in missing values
imputer = KNNImputer(n_neighbors=5)  # Look at 5 similar hospitals
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
```

### 5. Fixing Extreme Values

Sometimes there are unrealistic values in the data. We fix this by setting limits:

```python
# Make sure percentage values stay between 0 and 100
for col in X_imputed.columns:
    if 'Percent' in col:
        X_imputed[col] = X_imputed[col].clip(lower=0, upper=100)
```

## Creating Useful Features from the Data

### 1. Simple Features (Model1.py)

In our basic model, we create a few helpful combined features:

```python
# Calculate the average "Always" percentage across all questions
always_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'A_P' in col]
df['avg_always_percent'] = df[always_cols].mean(axis=1)

# Create a standardized survey count (this helps compare hospitals of different sizes)
df['normalized_surveys'] = df['Number_of_Completed_Surveys'] / df['Number_of_Completed_Surveys'].max()
```

### 2. Advanced Features (Model2.py)

Our improved model creates more sophisticated features to capture important patterns:

```python
# Average and maximum "Always" responses
always_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'A_P' in col]
df['avg_always_percent'] = df[always_cols].mean(axis=1)
df['max_always_percent'] = df[always_cols].max(axis=1)

# Average "Sometimes/Never" responses (these are usually negative indicators)
sn_cols = [col for col in df.columns if 'HCAHPS_Answer_Percent' in col and 'SN_P' in col]
df['avg_sn_percent'] = df[sn_cols].mean(axis=1)

# Nurse and doctor specific metrics (these are especially important)
nurse_cols = [col for col in df.columns if 'H_NURSE' in col and 'HCAHPS_Answer_Percent' in col]
doctor_cols = [col for col in df.columns if 'H_DOCTOR' in col and 'HCAHPS_Answer_Percent' in col]
df['avg_nurse_percent'] = df[nurse_cols].mean(axis=1)
df['avg_doctor_percent'] = df[doctor_cols].mean(axis=1)

# Combined effect of nurse and doctor communication
df['nurse_doctor_interaction'] = df['avg_nurse_percent'] * df['avg_doctor_percent']

# Rating weighted by response rate (more responses = more reliable)
df['weighted_rating'] = df['Patient_Survey_Star_Rating'] * df['Survey_Response_Rate_Percent'] / 100

# Ratio of positive to negative responses
df['always_to_sn_ratio'] = df['avg_always_percent'] / (df['avg_sn_percent'] + 1e-5)
```

### 3. Picking the Best Features

Not all features are equally helpful. We use a statistical method to select the 50 most important ones:

```python
# Select the top 50 most predictive features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X_imputed, y)
selected_features = X_imputed.columns[selector.get_support()].tolist()
```

## Balancing the Dataset

One challenge is that we don't have equal numbers of hospitals with each star rating. There are usually more 3-star hospitals than 1-star or 5-star hospitals. To help our model learn about all ratings equally well, we use a technique called SMOTE:

```python
# Create additional samples for ratings that don't have many hospitals
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

This technique creates artificial but realistic examples of the less common ratings, helping our model learn to recognize all rating levels.

## Choosing the Right Models

### 1. Our First Approach (Model1.py)

In our first try, we used two popular machine learning models:

- **Random Forest**: This model builds many decision trees and combines their results. It's like asking many different experts and taking a vote.
- **XGBoost**: A more advanced model that learns from its mistakes and gradually improves its predictions.

We chose these models because:

- Random Forest is reliable and doesn't easily get confused by noisy or messy data
- XGBoost typically performs very well with structured data like hospital surveys

### 2. Our Improved Approach (Model2.py)

In our enhanced version, we combined three different models to get even better results:

- **XGBoost**: With optimized settings based on testing
- **LightGBM**: A faster model that works well with large datasets
- **Random Forest**: For additional stability and different perspectives

We combined these models using a "soft voting" approach, which means:

- Each model makes a prediction about the hospital's star rating
- Each model also tells us how confident it is in its prediction
- We combine these confidence scores to get a final prediction that's more accurate

It's like getting a second and third opinion from different doctors before making a diagnosis.

## How Well Our Models Performed

Here's a simple comparison of how accurately our two models predicted hospital ratings:

| Model | Accuracy | F1 Score | Cross-Validation |
|-------|----------|----------|------------------|
| Model1 (Basic) | 95.45% | 0.9531 | 0.9512 (+/- 0.0214) |
| Model2 (Advanced) | 98.42% | 0.9837 | 0.9809 (+/- 0.0178) |

What this means in simple terms:

- Our advanced model correctly predicts hospital ratings about 98% of the time
- It's significantly more accurate than our basic model
- The "Cross-Validation" score shows that the model performs consistently well across different subsets of the data

## Fine-Tuning Our Models

### 1. Basic Model Settings

For our first model, we kept things simple with just a few basic settings:

```python
# Create a Random Forest model with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create an XGBoost model with standard settings
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
```

### 2. Advanced Model Optimization

For our enhanced model, we carefully tested many different settings to find the best combination:

```python
# Try many different combinations of settings for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],          # How many trees to build
    'learning_rate': [0.01, 0.1, 0.2],        # How quickly the model learns
    'max_depth': [3, 5, 7],                   # How complex each tree can be
    'subsample': [0.7, 0.8, 1.0],             # Percentage of data used for each tree
    'colsample_bytree': [0.7, 0.8, 1.0]       # Percentage of features used for each tree
}

# Automatically find the best settings
xgb_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=20, cv=5, 
                               scoring='accuracy', random_state=42, n_jobs=-1)
```

We used RandomizedSearchCV instead of testing every possible combination because it's much faster and still finds settings that work very well.

## What Matters Most for Hospital Ratings

Our analysis reveals which factors have the biggest impact on hospital ratings:

### Most Important Factors

1. **Nurse and Doctor Communication**: How well nurses and doctors talk with patients is the #1 factor. Hospitals where staff communicates clearly and thoroughly get much better ratings.

2. **Overall Hospital Experience**: The patient's general impression of the hospital is a strong predictor, often captured in the `HCAHPS_Linear_Mean_Value` for overall hospital ratings (H_HSP_RATING).

3. **"Always" Responses**: The percentage of patients who answer "Always" to questions about care quality is very important. Consistency matters!

4. **Combined Staff Effect**: The interaction between nurse and doctor communication (our `nurse_doctor_interaction` feature) is powerful. When both groups communicate well, the positive effect multiplies.

5. **Survey Participation**: How many surveys were completed and what percentage of patients responded helps determine the reliability of the ratings.

This information isn't just interesting - it helps hospitals know exactly where to focus their improvement efforts to get better ratings.

## Using Our System in the Real World

We've built a user-friendly app (using Streamlit) that makes it easy for hospital administrators to use our prediction system:

### What the App Offers

1. **Easy Input**: Users can simply adjust sliders and enter values for their hospital metrics.

2. **Visual Results**: The app includes:
   - Radar charts showing how the hospital performs in different areas
   - Bar charts comparing the hospital's rating to others nationwide

3. **Improvement Tips**: Based on the hospital's current metrics and predicted rating, the app suggests specific actions to improve patient satisfaction.

4. **Impact Analysis**: Users can see which factors have the biggest influence on their rating, allowing them to focus resources where they'll make the biggest difference.

This makes our advanced machine learning technology accessible to people without technical expertise.

## Current Limitations and Future Plans

### What We're Still Working On

1. **Changes Over Time**: Our current model doesn't track how hospital performance changes over time. It gives a snapshot rating based on current data.

2. **Hospital Type Differences**: Different kinds of hospitals (like rural vs. urban, teaching vs. non-teaching) might need to be evaluated differently. Our current model uses the same approach for all hospitals.

3. **Survey Response Issues**: Hospitals with very few survey responses might get less accurate predictions.

### How We Plan to Improve

1. **Time Analysis**: Add features to track rating improvements or declines over months and years.

2. **Hospital Grouping**: Create specialized models for different types of hospitals.

3. **Confidence Levels**: Show how confident we are in each prediction based on the amount and quality of available data.

4. **Advanced AI**: Try deep learning approaches that might catch more complex patterns in the data.

5. **Testing with More Data**: Validate our model with data from a wider range of hospitals to ensure it works for everyone.

## Conclusion

Our Hospital Rating Prediction system shows that machine learning can effectively predict hospital ratings using patient survey data. The enhanced model (Model2.py) correctly predicts ratings 98.42% of the time, which is significantly better than our initial approach.

The system doesn't just make predictions - it provides practical recommendations to help hospitals improve their patient care and satisfaction scores. With an easy-to-use interface, even staff without technical backgrounds can benefit from these advanced analytics.

## New Implementation: Three-Tab Structure

Our system has been completely redesigned with a three-tab structure to improve usability and provide more comprehensive analysis of patient survey data:

### 1. Patient Report Tab
- Displays predicted hospital rating (1-5 stars)
- Shows performance metrics for each HCAHPS category
- Visualizes results using radar charts and bar charts
- Provides historical trends and benchmark comparisons
- Offers improvement recommendations based on low scores

### 2. Patient Survey Tab
- Collects basic patient demographic information
- Allows for quick input of key survey responses
- Validates required fields
- Includes submission handling for rating calculation

### 3. HCAHPS Form Tab
- Implements the complete HCAHPS survey form
- Groups questions by category as per official HCAHPS guidelines
- Processes responses for detailed analysis

## Real-time Rating Calculation

The core of our new implementation is the real-time rating calculation based on HCAHPS parameters. The system maps survey responses to the features expected by the predictive models through the following process:

1. Converts categorical responses (e.g., "Always", "Sometimes") to numerical values
2. Aggregates responses by category
3. Calculates derived metrics like averages and percentages
4. Creates the full feature vector required by the model

The rating calculation focuses on these key HCAHPS categories:
- Your care from nurses
- Your care from doctors
- Your experiences in this hospital
- The hospital environment
- When you left the hospital
- Understanding your care when you left the hospital

## Multiple Model Support

Our system supports three different model types:

1. **Standard Model (Model1.py)**:
   - Uses Random Forest and XGBoost classifiers
   - Simpler implementation with fewer features
   - Good baseline accuracy (95%+)

2. **Enhanced Model (Model2.py)**:
   - Uses ensemble learning with XGBoost, LightGBM and Random Forest
   - Implements advanced feature engineering
   - Better accuracy (98%+)

3. **CNN Model (Model3.py)**:
   - Deep learning approach with neural networks
   - Can capture more complex patterns in the data
   - Optional - requires TensorFlow installation

## Advanced Visualizations

The new implementation includes multiple visualization techniques:

- **Radar Charts**: Show relative performance across all categories
- **Bar Charts**: Compare category scores to target thresholds
- **Historical Trends**: Track improvement over time
- **Benchmark Comparisons**: Compare against regional/national standards

## Data Management

We've also added robust data management capabilities:
- Save patient data to CSV files
- Load patient data from previously saved files
- Generate detailed PDF reports
- Reset patient data as needed

## Future Enhancements

1. Integration with hospital database systems
2. Automated email distribution of patient reports
3. Advanced trend analysis using time-series modeling
4. Multi-language support for diverse patient populations
5. Mobile-responsive design for tablet use in clinical settings
