import streamlit as st
import pickle as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.graph_objects as go
import numpy as np
def get_clean_data():
    df= pd.read_csv("https://raw.githubusercontent.com/Abhishekpn09/Breast-cancer-predictor/master/.devcontainer/Cancerdataset.csv")
    df=df.drop(['Unnamed: 32','id'], axis=1)
    df['diagnosis'].replace({'M':1,'B':0},inplace=True)
    return df

def add_sidebar():
    st.sidebar.header('Cell Nuclei Details')
    df = get_clean_data()
    
    slider_labels = [
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
    input_dict ={}
    for label, key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(df[key].max()),
            value=float(df[key].max())
        )
    return input_dict

def get_scaled_values(input_dict):
    df=get_clean_data()
    
    x=df.drop(['diagnosis'], axis=1)
    
    scaled_dict ={}
    
    for key, value in input_dict.items():
        max_val=x[key].max()
        min_val=x[key].min()
        scaled_value =(value-min_val)/(max_val -min_val)
        scaled_dict[key]=scaled_value
    return scaled_dict
def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

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
  
  return fig

def train_model():
    df = get_clean_data()
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Train model when app starts
model = train_model()

def add_predictions(input_data):
    scaled_input = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(scaled_input)

    st.subheader("Cell cluster prediction")
    if prediction[0] == 0:
        st.write("Benign")
    else:
        st.write("Malignant")

    st.write("Probability of being benign:", model.predict_proba(scaled_input)[0][0])
    st.write("Probability of being malignant:", model.predict_proba(scaled_input)[0][1])
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")





def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female -doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data=add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    col1, col2 = st.columns([4,1])

    with col1:  
        radar_chart=get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)



if __name__ == '__main__':
    main()


