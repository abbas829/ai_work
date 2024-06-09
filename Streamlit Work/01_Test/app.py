import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Streamlit configuration
st.set_page_config(
    page_title="Comprehensive Data Dashboard",
    page_icon="📊",
    layout="wide"
)

# Function to read data from a CSV file
@st.experimental_memo
def load_data_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

# Function to read data from an Excel file
@st.experimental_memo
def load_data_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)

# Function to read data from an SQL database
@st.experimental_memo
def load_data_sql(connection_string, query) -> pd.DataFrame:
    engine = create_engine(connection_string)
    return pd.read_sql(query, engine)

# Function to encode categorical columns
def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Function to clean the data
def clean_data(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Drop rows with missing values
    df = df.dropna()
    return df

# Function for feature scaling
def scale_features(df):
    # Select only numeric columns for scaling
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_numeric = df[numeric_columns]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)
    
    # Create a DataFrame with scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=df_numeric.columns)
    return df_scaled

# Dashboard header
st.title("Comprehensive Data Dashboard")

# Data source selection
data_source = st.sidebar.selectbox("Select data source", ["CSV", "Excel", "SQL Database"])

# File uploader or SQL input based on data source
if data_source == "CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data_csv(uploaded_file)

elif data_source == "Excel":
    uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")
    if uploaded_file is not None:
        df = load_data_excel(uploaded_file)

elif data_source == "SQL Database":
    connection_string = st.text_input("Enter SQL connection string")
    query = st.text_area("Enter SQL query")
    if st.button("Load Data"):
        if connection_string and query:
            df = load_data_sql(connection_string, query)
        else:
            st.error("Please provide both connection string and query.")

# Display and process the data if loaded
if 'df' in locals():
    # Display the dataset
    st.subheader("Dataset")
    st.write(df)
    
    # Data cleaning options
    if st.checkbox("Perform Data Cleaning"):
        df_cleaned = clean_data(df)
        st.subheader("Cleaned Dataset")
        st.write(df_cleaned)

    # Feature scaling options
    if st.checkbox("Perform Feature Scaling"):
        df_scaled = scale_features(df)
        st.subheader("Scaled Features")
        st.write(df_scaled)
    
    # Select columns for plotting
    columns = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis column", columns)
    y_axis = st.selectbox("Select Y-axis column", columns)
    
    # Tabs for different types of graphs
    graph_type = st.radio("Select Graph Type", ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Box Plot"])
    if graph_type == "Scatter Plot":
        fig = px.scatter(df, x=x_axis, y=y_axis)
    elif graph_type == "Line Plot":
        fig = px.line(df, x=x_axis, y=y_axis)
    elif graph_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis)
    elif graph_type == "Histogram":
        fig = px.histogram(df, x=x_axis)
    elif graph_type == "Box Plot":
        fig = px.box(df, x=x_axis, y=y_axis)

    # Additional graph types can be added here
    
    st.plotly_chart(fig)

    # Add other components as needed
    st.markdown("### Additional Insights")
    st.write("Descriptive Statistics:")
    st.write(df.describe())

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        df_encoded = encode_categorical(df.copy())
        corr = df_encoded.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig_corr)

    # Footer
    st.markdown(
        """
        ---
        Created by Tasswar Abbas  
        Email: abbas829@gmail.com  
        GitHub: [github.com/abbas829](https://github.com/abbas829)
        """
    )
else:
    st.info("Please upload a file or connect to a database to proceed.")
