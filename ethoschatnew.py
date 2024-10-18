import streamlit as st
from openai import OpenAI
import snowflake.connector
import pandas as pd
import json
import traceback

# Initialize session state variables
if "snowflake_connected" not in st.session_state:
    st.session_state.snowflake_connected = False
if "snowflake_params" not in st.session_state:
    st.session_state.snowflake_params = {}
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

st.title("❄️ Snowflake Account Data Assistant")

# Snowflake connection parameters
if not st.session_state.snowflake_connected:
    with st.sidebar:
        st.header("Snowflake Connection")
        SNOWFLAKE_ACCOUNT = st.text_input("Account", value="au02318.eu-west-2.aws")
        SNOWFLAKE_USER = st.text_input("Username", value="salesmachinePOC")
        SNOWFLAKE_PASSWORD = st.text_input("Password", type="password")
        SNOWFLAKE_DATABASE = st.text_input("Database", value="FIRSTDB")
        SNOWFLAKE_SCHEMA = st.text_input("Schema", value="PUBLIC")
        OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
        
        if st.button("Connect"):
            try:
                # Test connection
                conn = snowflake.connector.connect(
                    account=SNOWFLAKE_ACCOUNT,
                    user=SNOWFLAKE_USER,
                    password=SNOWFLAKE_PASSWORD,
                    database=SNOWFLAKE_DATABASE,
                    schema=SNOWFLAKE_SCHEMA
                )
                conn.cursor().execute("SELECT 1")
                conn.close()
                st.session_state.snowflake_connected = True
                st.session_state.snowflake_params = {
                    "account": SNOWFLAKE_ACCOUNT,
                    "user": SNOWFLAKE_USER,
                    "password": SNOWFLAKE_PASSWORD,
                    "database": SNOWFLAKE_DATABASE,
                    "schema": SNOWFLAKE_SCHEMA
                }
                st.session_state.openai_api_key = OPENAI_API_KEY
                st.success("Connected successfully!")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
                st.error(traceback.format_exc())

if st.session_state.snowflake_connected:
    # Function to execute SQL query with error handling
    def execute_sql_query(query):
        conn = snowflake.connector.connect(**st.session_state.snowflake_params)
        try:
            cur = conn.cursor()
            cur.execute(query)
            results = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            df = pd.DataFrame(results, columns=columns)
            return df
        except Exception as e:
            st.error(f"Error executing query: {query}")
            st.error(f"Error message: {str(e)}")
            st.error(traceback.format_exc())
            return pd.DataFrame()  # Return an empty DataFrame on error
        finally:
            conn.close()

    # Function to get table schema
    def get_table_schema():
        query = "DESCRIBE TABLE ACCOUNT"
        schema = execute_sql_query(query)
        if schema.empty:
            st.warning("The ACCOUNT table schema is empty or the table does not exist.")
        return schema

    # Function to get data summary
    def get_data_summary():
        summary = {}
        df = execute_sql_query("SELECT * FROM ACCOUNT LIMIT 1000")  # Limit to 1000 rows for performance
        if df.empty:
            return {"error": "The ACCOUNT table is empty or no data could be retrieved."}
        
        summary['total_rows'] = len(df)
        summary['columns'] = df.columns.tolist()
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if not df[col].isnull().all():
                    summary[f'{col}_avg'] = float(df[col].mean())
                    summary[f'{col}_max'] = float(df[col].max())
                    summary[f'{col}_min'] = float(df[col].min())
                else:
                    summary[f'{col}_avg'] = None
                    summary[f'{col}_max'] = None
                    summary[f'{col}_min'] = None
            elif df[col].dtype == 'object':
                non_null_values = df[col].dropna()
                summary[f'{col}_unique'] = int(non_null_values.nunique())
                if not non_null_values.empty:
                    top_value = non_null_values.value_counts().index[0]
                    summary[f'{col}_top'] = str(top_value)
                else:
                    summary[f'{col}_top'] = None
        return summary

    def get_relevant_data(question, schema, summary):
        if "error" in summary:
            return {"error": summary["error"]}
        
        try:
            # Summarize schema
            schema_summary = ", ".join([f"{row['name']} ({row['type']})" for _, row in schema.iterrows()])
            
            # Summarize data summary
            data_summary = {
                "total_rows": summary["total_rows"],
                "columns": summary["columns"]
            }
            
            # Truncate summaries if they're too long
            max_length = 1000  # Adjust this value as needed
            if len(schema_summary) > max_length:
                schema_summary = schema_summary[:max_length] + "..."
            if len(json.dumps(data_summary)) > max_length:
                data_summary["columns"] = data_summary["columns"][:10]  # Limit to first 10 columns
                data_summary["note"] = "Summary truncated due to length"
            
            openai_client = OpenAI(api_key=st.session_state.openai_api_key)
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Based on the user's question, determine which columns from the ACCOUNT table are relevant. Return only the column names as a comma-separated list."},
                    {"role": "user", "content": f"Question: {question}\nSchema: {schema_summary}\nSummary: {json.dumps(data_summary)}"}
                ]
            )
            relevant_columns = response.choices[0].message.content.split(',')
            relevant_columns = [col.strip() for col in relevant_columns if col.strip() in summary['columns']]
            
            relevant_data = {col: summary.get(f'{col}_avg') or summary.get(f'{col}_unique') or summary.get(f'{col}_top') for col in relevant_columns}
            return relevant_data
        except Exception as e:
            st.error(f"Error in get_relevant_data: {str(e)}")
            st.error(traceback.format_exc())
            return {"error": "Failed to process relevant data"}
        
    def get_openai_response(question, relevant_data, summary):
        if "error" in relevant_data:
            return relevant_data["error"]
        
        try:
            openai_client = OpenAI(api_key=st.session_state.openai_api_key)
            context = f"ACCOUNT table summary: {json.dumps(summary)}\nRelevant data: {json.dumps(relevant_data)}"
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a data analyst. Answer questions about the ACCOUNT data using this context: {context}"},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in get_openai_response: {str(e)}")
            st.error(traceback.format_exc())
            return "Failed to generate a response"


    # Main interface
    st.header("Ask about Account Data")
    user_question = st.text_input("Enter your question about the ACCOUNT data:")

    if user_question:
        with st.spinner("Analyzing data..."):
            try:
                schema = get_table_schema()
                if not schema.empty:
                    st.write("Schema retrieved successfully")
                    st.write("Fetching data summary...")
                    summary = get_data_summary()
                    st.write("Data summary created")
                    
                    if "error" not in summary:
                        st.write("Processing relevant data...")
                        relevant_data = get_relevant_data(user_question, schema, summary)
                        st.write("Relevant data processed")
                        st.write("Generating OpenAI response...")
                        answer = get_openai_response(user_question, relevant_data, summary)
                        
                        st.subheader("Answer:")
                        st.write(answer)
                    else:
                        st.error(summary["error"])
                else:
                    st.error("Failed to retrieve schema for ACCOUNT table")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error(traceback.format_exc())

    # Optional: Display schema and summary
    if st.checkbox("Show ACCOUNT table information"):
        try:
            st.subheader("Table Schema")
            schema = get_table_schema()
            if not schema.empty:
                st.dataframe(schema)
            else:
                st.warning("The ACCOUNT table schema is empty or could not be retrieved.")

            st.subheader("Data Summary")
            summary = get_data_summary()
            if "error" not in summary:
                st.json(summary)
            else:
                st.warning(summary["error"])
        except Exception as e:
            st.error(f"An error occurred while fetching table information: {str(e)}")
            st.error(traceback.format_exc())

else:
    st.warning("Please connect to Snowflake using the sidebar to start querying data.")

# Display Snowflake Connector version
st.sidebar.write(f"Snowflake Connector Python Version: {snowflake.connector.__version__}")