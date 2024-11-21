import streamlit as st
import pandas as pd
import altair as alt
import kagglehub
import json

# Title and description
st.title("Yelp Dataset Analysis App")
st.write("This app demonstrates data visualization with the Yelp dataset using Streamlit and Altair.")

# Load data
@st.cache_data
def load_data():
    # Download the Yelp dataset from Kaggle Hub
    path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
    
    # Load the 'yelp_academic_dataset_checkin.json' file
    data_file = open(f"{path}/yelp_academic_dataset_checkin.json")
    data = [json.loads(line) for line in data_file]
    data_file.close()
    
    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Debugging step: Show all columns and their types
    st.write("### Dataset Columns and Types:")
    st.write(df.dtypes)

    # Handle the 'date' column (convert to a count of timestamps per business_id)
    if 'date' in df.columns:
        df['date_count'] = df['date'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    
    # Return the cleaned dataset and numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return df, numeric_columns

# Load the data and numeric columns
data, numeric_columns = load_data()

# Display the first 10 rows of the data
st.write("### Sample Dataset:")
st.dataframe(data.head(10))

# Show the list of columns
columns = data.columns.tolist()

# Display available numeric columns for Y-axis selection
st.write("### Debug Information:")
st.write("Available Columns:", columns)
st.write("Numeric Columns for Y-axis:", numeric_columns)

# Handle empty dataset
if len(columns) == 0:
    st.error("The dataset has no valid columns to visualize.")
else:
    # Allow the user to select the X-axis column
    x_col = st.selectbox("Select X-axis:", options=columns)
    
    # Only show the Y-axis dropdown if there are numeric columns available
    if numeric_columns:
        y_col = st.selectbox("Select Y-axis (numeric):", options=numeric_columns)
    else:
        st.warning("No numeric columns available for Y-axis.")
        y_col = None  # Disable Y-axis if no numeric columns are found

    # Handle missing or invalid data
    st.write("### Visualization:")
    try:
        if x_col and y_col:
            clean_data = data[[x_col, y_col]].dropna()

            if clean_data.empty:
                st.warning("No data available after cleaning. Please adjust your selection.")
            else:
                max_rows = 5000  # Limit to 5000 rows for visualization
                if len(clean_data) > max_rows:
                    st.warning(f"Dataset is too large; only visualizing the first {max_rows} rows.")
                    clean_data = clean_data.head(max_rows)

                # Create the Altair chart for the selected columns
                chart = alt.Chart(clean_data).mark_circle(size=60).encode(
                    x=alt.X(x_col, title=x_col),
                    y=alt.Y(y_col, title=y_col),
                    tooltip=[x_col, y_col]
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Please select valid columns for both X and Y axes.")
    except KeyError as e:
        st.error(f"KeyError: {e}. Please check the column names.")
    except Exception as e:
        st.error(f"Error generating chart: {e}")