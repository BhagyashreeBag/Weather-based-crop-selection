import numpy as np
import pandas as pd
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("crop_prediction_model_one.csv")

converts_dict = {
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Rainfall': 'rainfall',
    'ph': 'ph'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def scatterPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def main():
    html_temp_vis = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize Soil Properties </h2>
    </div>
    """

    html_temp_pred = """
    <div style="background-color:#025246 ;padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """

    st.sidebar.title("Select One")
    select_type = st.sidebar.radio("", ('Graph', 'Predict Your Crop'))


    if select_type == 'Graph':
        st.markdown(html_temp_vis, unsafe_allow_html=True)
        plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
        st.subheader("Relation between features")

        # Plot!

        x = ""
        y = ""

        if plot_type == 'Bar Plot':
            x = 'label'
            y = st.selectbox("Select a feature to compare between crops",
                ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
        if plot_type == 'Scatter Plot':
            x = st.selectbox("Select a property for 'X' axis",
                ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
            y = st.selectbox("Select a property for 'Y' axis",
                ('Nitrogen', 'Phosphorus', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
        if plot_type == 'Box Plot':
            x = "label"
            y = st.selectbox("Select a feature",
                ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))

        if st.button("Visulaize"):
            if plot_type == 'Bar Plot':
                y = converts_dict[y]
                barPlotDrawer(x, y)
            if plot_type == 'Scatter Plot':
                x = converts_dict[x]
                y = converts_dict[y]
                scatterPlotDrawer(x, y)
            if plot_type == 'Box Plot':
                y = converts_dict[y]
                boxPlotDrawer(x, y)
    
    if select_type == "Predict Your Crop":
        st.markdown(html_temp_pred, unsafe_allow_html=True)
        st.header("To predict your crop give values")
        st.subheader("Drag to Give Values")
        n = st.slider('Nitrogen', 0, 140)
        p = st.slider('Phosphorus', 5, 145)
        k = st.slider('Potassium', 5, 205)
        temperature = st.slider('Temperature', 8.83, 43.68)
        humidity = st.slider('Humidity', 14.26, 99.98)
        ph = st.slider('pH', 3.50, 9.94)
        rainfall = st.slider('Rainfall', 20.21, 298.56)
        
        if st.button("Predict your crop"):
            output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
            res = "“"+ output.capitalize() + "”"
            st.success('The most suitable crop for your field is {}'.format(res))

if __name__=='__main__':
    main()
    
