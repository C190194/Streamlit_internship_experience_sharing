import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# table:
# Ori_ResNet18_df= pd.read_csv('Original_ResNet18_Archi.csv')
# st.table(Ori_ResNet18_df)

# table with colored columns:
# result_df = pd.read_csv('Results.csv')
# col_ref = {'Baseline 1:': 'background-color: #ffec8c', 
#         'Autoencoder': 'background-color: #ffec8c', 
#         'Baseline 2:':'background-color: #c2f5ff',
#         'MobileNetV2':'background-color: #c2f5ff'}
# st.table(result_df.style.apply(lambda x: pd.DataFrame(col_ref, \
#     index=result_df.index, columns=result_df.columns).fillna(''), axis=None))

# picture:
# ResNet_archi_AdaCos_image = Image.open('ResNet_archi_AdaCos.png')
# st.image(ResNet_archi_AdaCos_image)

# column:
# col1, col2, col3 = st.columns(3)
# with col1:

st.set_page_config(layout="wide")

# st.title("An Overview of the Anomalous Sound Detection Project") 

select_slide = st.sidebar.selectbox(
    "Which slide would you like to navigate to?",
    ("Project Overview", "Pytorch & Tensorflow", "Streamlit", "Docker")
)

if select_slide == "Project Overview": 

    st.header("An Overview of the Anomalous Sound Detection Project") 
    st.write(" ")

    st.write("**Aim:** To build a Python program with Pytorch that can detect \
            anomalous sounds from a shifted domain")
    st.write("The topic was taken from [DCASE Challenge 2021 (Task 2)](https://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds#description)")
    st.write(" ")

    st.write("**Challenges:** ")
    st.write("1. All the sounds for training were normal data")
    st.write("2. Most of the training data was collected from machines working \
            in the source domain")
    st.write(" ")

    st.write("**Application:** To help factories to detect machine failures \
            in both source and target domains and reduce the cost of recording \
            sounds for training the model.")
    st.write(" ")

    st.write("The **general process** of the solution is shown here:")
    st.write("1. Transform the sound file into a log-mel-spectrogram, which is a matrix")
    st.write("2. Train models to generate a score for each sound file")
    st.write("3. Make classification based on the score.")
    st.write(" ")

    st.write("**Current progress:**")
    st.write("1. Tested various unsupervised anomaly detection methods from libraries, such as Scikit-learn and PyOD")
    st.write("2. Implementing the solution which got the 3rd place in the competition")
    st.write(" ")

    

elif select_slide == "Pytorch & Tensorflow":
    pytorch_logo = Image.open('pytorch_logo.png')
    tensorflow_logo = Image.open('TensorFlow_logo.svg.png')

    st.header("Libraries for Building Neural Networks") 

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pytorch_logo)
    with col2:
        st.image(tensorflow_logo)
    
    st.write("After following the tutorials in their official documents \
        ([Pytorch](https://pytorch.org/tutorials/) & [Tensorflow](https://pytorch.org/tutorials/)), \
        I have become more familiar with their data loading, training and testing stages.")
    
elif select_slide == "Streamlit":

    st.subheader("Streamlit") 
    st.write(" ")

    st.write("Most of the presentation slides during internship at NCS will be made in Streamlit")
    st.write(" ")

    st.write("**Advantages:**")
    st.write("1. Interactive widgets can be easily added to slides")
    st.write("2. Numbers and graphs can dynamically change according to your inputs")
    st.write(" ")

    st.write("**Widget example:**")
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'''
            f(x) = sin(kx + w\pi)
            ''')
        k = st.select_slider(
            'Choose a value for k',
            options=[-0.5, 0, 0.5, 1, 1.5, 2],
            value=1)
        w = st.slider('Choose a value for w', min_value=0.0, max_value=2.0, step=0.05)
        if k == 0 or k == 1:
            s_k = ""
        else:
            s_k = str(k)
        if w == 0 or w == 1:
            s_w = ""
        else:
            s_w = str(w)
        str = "f(x) = sin(" + s_k + "x + " + s_w + "\pi)"
        st.latex(str)

        # 100 linearly spaced numbers
        x = np.linspace(-np.pi,np.pi,100)

        # the function, which is y = sin(x) here
        y = np.sin(k * x + w * np.pi)

        # setting the axes at the centre
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # plot the function
        plt.plot(x,y, 'b-')
        st.pyplot(fig)

    st.write(" ")

    st.write("Official tutorials can be found [here](https://docs.streamlit.io/)")

elif select_slide == "Docker":
    docker_logo = Image.open('docker_logo.png')
    env_pic = Image.open('env.png')
    dockerfile_pic = Image.open('dockerfile.png')
    command1_pic = Image.open('command1.png')
    command2_pic = Image.open('command2.png')

    st.subheader("Docker") 
    st.write(" ")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Docker allows us to deploy and run our codes on \
            different machines with the same package dependencies.")
        st.write(" ")

        st.write("**Step 1:** Output the environment.yml file")
        st.image(env_pic, width = 250)
        st.write(" ")

        st.write("**Step 2:** Add a Dockerfile into the project folder")
        st.image(dockerfile_pic, width = 600)
        st.write(" ")

        st.write("**Step 3:** Run 2 lines of command and wait for Docker to \
            set up the environment, download all the packages required, \
            and run the code for you")
        st.image(command1_pic)
        st.image(command2_pic)

    with col2:
        st.image(docker_logo)


    st.write(" ")



