import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.graph_objects as go

# Load the cleaned dataset
df = pd.read_csv("model/data_cleaned.csv")

# Function to load the sidebar
def load_sidebar():
    # Set the title of the sidebar
    st.sidebar.header("Cell Nuclei Measurements")

    # Define the labels
    labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Create a dictionary to store the sliders
    sidebar_sliders = {}

    # Add the sliders to the sidebar
    for label, key in labels:
        sidebar_sliders[key] = st.sidebar.slider(label, float(0), float(df[key].max()), float(df[key].mean()))

    # Return the sliders
    return sidebar_sliders

# Function to load the scaled values for the radar chart
def get_scaled_values(input_sidebar):
    # Drop the target variable
    X = df.drop(["diagnosis"], axis=1)

    # Create a dictionary to store the scaled values
    scaled_dict = {}

    # Scale the values
    for key, value in input_sidebar.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value

    # Return the scaled values
    return scaled_dict

# Function to load the radar chart
def load_radar_chart(input_sidebar):
    # Get the scaled values
    input_data = get_scaled_values(input_sidebar)

    # Define the categories
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    # Create the radar chart
    fig = go.Figure()

    # Add the traces
    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
        input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
        input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
        input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
        input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
        input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
        input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
        input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
        input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
        input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )

    # Return the radar chart
    return fig

# Function to load the prediction
def load_prediction(input_sidebar):
    # Load the model and the scaler
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # Initialize the input data
    input_array = np.array(list(input_sidebar.values())).reshape(1, -1)

    # Scale the input data
    input_df = pd.DataFrame(input_array, columns=df.drop(["diagnosis"], axis=1).columns)
    input_scaled = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(input_scaled)

    # Display the prediction
    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is predicted to be:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    # Display the probability
    st.write("Probability of being benign:", model.predict_proba(input_scaled)[0][0])
    st.write("Probability of being malicious:", model.predict_proba(input_scaled)[0][1])

    # Display the disclaimer
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

# Main function
def main():
    # Set the title, icon and layout of the web app
    st.set_page_config(page_title="CancerGuardian", page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")
    
    # Set the title of the web app
    st.title("CancerGuardian")

    # Load the CSS file
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Add the Sidebar
    input_sidebar = load_sidebar()

    # Container for the main content
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

    # Create two columns with a ratio of 4:1
    col1, col2 = st.columns([4, 1])

    # Load the radar chart in the first column
    with col1:
        radar_chart = load_radar_chart(input_sidebar)
        st.plotly_chart(radar_chart)

    # Load the prediction in the second column
    with col2:
        load_prediction(input_sidebar)

# Entry point of the script
if __name__ == "__main__":
    main()
