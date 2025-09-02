import streamlit as st
import pandas as pd
import numpy as np
import pylatex

st.title("CSV Uploader and Viewer")



uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        dataframe = pd.read_csv(uploaded_file)

        st.write("File uploaded successfully! Here's a preview:")
        # Display the DataFrame in Streamlit
        st.dataframe(dataframe)

        # You can now perform further operations on the 'dataframe'
        # For example, display basic statistics:
        st.subheader("Basic Statistics:")
        st.write(dataframe.describe())
        st.write(f"""Here is more text\n
                 and the contents of cell (0,0) {dataframe.iloc[0, 0]}\n
                and here is pi {np.pi}"
        """)
        summary = dataframe.describe()

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

