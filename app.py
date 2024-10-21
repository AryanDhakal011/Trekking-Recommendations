# Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import numpy as np
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

# Load the dataset
file_path = 'data/Nepali_Trekking_Data.csv'  
trekking_data = pd.read_csv(file_path)

# Display raw dataset (optional)
st.title("Trekking Dataset Overview")
st.write("Here is the raw trekking dataset:")
st.dataframe(trekking_data)

# Preprocessing: Dropping unnecessary columns
columns_to_drop = [
    'Id', 'Trip Grade', 'Date of Travel', 'Sex', 'Regional code', 
    'Country', 'Weather Conditions', 'Trekking Group Size', 'Guide/No Guide', 
    'Equipment Used', 'Purpose of Travel', 'Health Incidents', 'Review/Satisfaction',
    'GraduateOrNot', 'FrequentFlyer', 'Employment Type', 'AnnualIncome'
]
df_cleaned = trekking_data.drop(columns=columns_to_drop)

# Displaying the size of the dataset after dropping columns
st.write("Size of dataset after dropping unnecessary columns:", df_cleaned.shape)

# Cleaning Cost column (removing '$', 'USD', and converting to numeric)
if 'Cost' in df_cleaned.columns:
    df_cleaned['Cost'] = df_cleaned['Cost'].replace({'\n': '', 'USD': '', '\$': '', ',': ''}, regex=True).astype(float)

# Cleaning Max Altitude column (removing 'm' and ',' and converting to numeric)
if 'Max Altitude' in df_cleaned.columns:
    df_cleaned['Max Altitude'] = df_cleaned['Max Altitude'].replace({' m': '', ',': ''}, regex=True)
    df_cleaned['Max Altitude'] = pd.to_numeric(df_cleaned['Max Altitude'], errors='coerce')
    df_cleaned['Max Altitude'] = df_cleaned['Max Altitude'].fillna(df_cleaned['Max Altitude'].median())

# Handling missing values in Fitness Level
if 'Fitness Level' in df_cleaned.columns:
    df_cleaned['Fitness Level'] = df_cleaned['Fitness Level'].fillna('Beginner')  # Fill NaNs with a placeholder
    label_enc = LabelEncoder()  # Instantiate the encoder
    df_cleaned['Fitness Level'] = label_enc.fit_transform(df_cleaned['Fitness Level'])  # Encode fitness levels

# Clean Accommodation column by removing "Hotel/" prefix
if 'Accommodation' in df_cleaned.columns:
    df_cleaned['Accommodation'] = df_cleaned['Accommodation'].str.replace('Hotel/', '', regex=False)

# Outlier Handling
z_scores = np.abs(stats.zscore(df_cleaned[['Cost', 'Max Altitude']].select_dtypes(include=[np.number])))
threshold = 3
df_cleaned = df_cleaned[(z_scores < threshold).all(axis=1)]

# Display the size of the dataset after outlier removal
st.write("Size of dataset after outlier removal:", df_cleaned.shape)

# Feature Scaling
scaler = StandardScaler()
df_cleaned[['Cost', 'Max Altitude']] = scaler.fit_transform(df_cleaned[['Cost', 'Max Altitude']])

# Preparing data for Random Forest
X = df_cleaned.drop('Cost', axis=1)
y = df_cleaned['Cost']

# Drop any rows with NaN values from features and ensure the target variable matches
X = X.dropna()
y = y[X.index]

# Display the remaining data sizes
st.write("Size of features after NaN removal:", X.shape)
st.write("Size of target variable after NaN removal:", y.shape)

# Ensure all features are numeric
X = X.select_dtypes(include=[np.number])


if X.shape[0] == 0 or y.shape[0] == 0:
    st.error("No samples left after cleaning. Please check the data preprocessing steps.")
else:
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Random Forest Regressor
    start_time_training = time.time()  
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time_training  

    # Making predictions
    start_time_prediction = time.time()  
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time_prediction  

    # Evaluating the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Displaying evaluation metrics
    st.header("Model Evaluation Metrics:")
    st.write(f"**Mean Absolute Error (MAE):** {mae}")
    st.write(f"**Mean Squared Error (MSE):** {mse}")

# Display cleaned and scaled dataset
st.header("Here is the cleaned dataset (for reference):")
st.dataframe(trekking_data)

# Linear Regression for Seasonal Demand Fluctuations
X_reg = df_cleaned[['Max Altitude']]
y_reg = df_cleaned['Cost']

# Ensure there are no NaN values
X_reg = X_reg.dropna()
y_reg = y_reg[X_reg.index]

# Splitting the dataset 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Training the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)

# Making predictions
y_pred_reg = lin_reg.predict(X_test_reg)

# Evaluating the regression model
mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)

# Displaying regression evaluation metrics
st.header("Linear Regression Model Evaluation Metrics:")
st.write(f"**Mean Absolute Error (MAE):** {mae_reg}")
st.write(f"**Mean Squared Error (MSE):** {mse_reg}")

# Plotting the results with the regression line
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values
plt.scatter(X_test_reg['Max Altitude'], y_test_reg, color='blue', label='Actual Demand')
plt.scatter(X_test_reg['Max Altitude'], y_pred_reg, color='red', label='Predicted Demand')

# Generate points for the regression line
x_range = np.linspace(X_test_reg['Max Altitude'].min(), X_test_reg['Max Altitude'].max(), 100).reshape(-1, 1)
# Predict using the regression model
y_range = lin_reg.predict(x_range)

# Plotting the regression line
plt.plot(x_range, y_range, color='green', label='Regression Line', linewidth=2)

plt.xlabel('Max Altitude')
plt.ylabel('Cost')
plt.title('Linear Regression: Actual vs Predicted Demand')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# K-Means Clustering Implementation
st.write("### K-Means Clustering on Trekking Data")

# Ensure only numeric data for K-Means
numeric_data = df_cleaned.select_dtypes(include=[np.number])

# Elbow Method to find the optimal value of k
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(numeric_data)  # Fit on only numeric data
    inertia.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
st.pyplot(plt)

# Asking user to input k value (based on elbow method)
optimal_k = st.number_input("Choose the optimal number of clusters (k) from the elbow graph:", min_value=1, max_value=10, value=3)

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(numeric_data)

# Plotting the clusters
plt.figure(figsize=(10, 6))
for i in range(optimal_k):
    cluster_data = df_cleaned[df_cleaned['Cluster'] == i]
    plt.scatter(cluster_data['Max Altitude'], cluster_data['Cost'], label=f'Cluster {i}')

plt.title(f'K-Means Clustering (k={optimal_k})')
plt.xlabel('Max Altitude')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)
st.pyplot(plt)

#bar graph
st.header("Fitness Level Distribution by Age")
# Define age categories
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40']
df_cleaned['Age Category'] = pd.cut(df_cleaned['Age'], bins=bins, labels=labels, right=False)

# Create a DataFrame for counts of fitness levels by age category
fitness_distribution = df_cleaned.groupby(['Age Category', 'Fitness Level']).size().unstack(fill_value=0)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
fitness_distribution.plot(kind='bar', stacked=True)

plt.title('Fitness Level Distribution by Age Categories')
plt.xlabel('Age Categories')
plt.ylabel('Count of Fitness Levels')
plt.xticks(rotation=0)
plt.legend(title='Fitness Level')
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)

# User Input for Trek Recommendation
st.header("Find Your Perfect Trekking Adventure!")

fitness_level = st.selectbox("Select Fitness Level:", options=['Beginner', 'Intermediate', 'Advanced'])
cost_input = st.number_input("Enter your budget (in $)", step=100)

# Filtering the dataset based on user input
if st.button("Get Recommendations"):
    if cost_input <= 449:
        st.warning("No treks available under this budget.")
    else:
        fitness_level_encoded = label_enc.transform([fitness_level])[0]
        
        recommendations = df_cleaned[
            (df_cleaned['Fitness Level'] == fitness_level_encoded) &
            (df_cleaned['Cost'] <= cost_input)
        ]
        
        # Limit to first 10 matches and format output
        if not recommendations.empty:
            # Show only the required columns with formatted values
            recommendations_display = recommendations[['Trek', 'Cost', 'Max Altitude', 'Accommodation', 'Fitness Level', 'Best Travel Time', 'Time']]
            
            # Display the original values instead of scaled values
            recommendations_display['Max Altitude'] = trekking_data['Max Altitude'].iloc[recommendations.index].astype(str) 
            recommendations_display['Cost'] = '$' + trekking_data['Cost'].iloc[recommendations.index].replace({'\$': '', 'USD': '', ',': ''}, regex=True) + ' USD'
            recommendations_display['Fitness Level'] = label_enc.inverse_transform(recommendations_display['Fitness Level'].astype(int))

            # Display the top 10 recommendations
            st.subheader("Here are your Recommended Treks:")
            st.dataframe(recommendations_display.head(10))  
        else:
            st.write("No treks found for your selection.")

#Checking Latency 

# For loading Dataset 
start_time_data_loading = time.time()
# End timer for data loading and compute latency
end_time_data_loading = time.time()
#calculating latency for loading dataset 
data_loading_latency = end_time_data_loading - start_time_data_loading

#For Preprocessing steps 
start_time_preprocessing = time.time()
end_time_preprocessing = time.time()
data_preprocessing_latency = end_time_preprocessing - start_time_preprocessing


start_time = time.time()  
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
train_time = time.time() - start_time 

start_time = time.time()  
y_pred = model.predict(X_test)
predict_time = time.time() - start_time 
st.write("### Checking Model Latency")
st.write(f"**Data Loading Latency:** {data_loading_latency:.6f} seconds")
st.write(f"**Data Preprocessing Latency:** {data_preprocessing_latency:.6f} seconds")
st.write(f"**Training Latency:** {train_time:.4f} seconds")
st.write(f"**Prediction Latency:** {predict_time:.4f} seconds")
