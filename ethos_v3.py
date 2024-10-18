import streamlit as st
from openai import OpenAI
import snowflake.connector
import pandas as pd
import json
import traceback
from collections import Counter
from datetime import datetime
import numpy as np
from datetime import datetime, date

# Initialize session state variables
if "snowflake_connected" not in st.session_state:
    st.session_state.snowflake_connected = False
if "snowflake_params" not in st.session_state:
    st.session_state.snowflake_params = {}
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "segmentation_data" not in st.session_state:
    st.session_state.segmentation_data = None

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
        return pd.DataFrame()
    finally:
        conn.close()

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, pd.Timestamp, date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

def safe_numeric_convert(df, column):
    try:
        return pd.to_numeric(df[column], errors='coerce')
    except:
        return pd.Series([np.nan] * len(df))

def calculate_age(born):
    today = datetime.now()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def get_table_info():
    try:
        df = execute_sql_query("SELECT * FROM ACCOUNT LIMIT 1")
        columns = df.columns.tolist()
        column_types = df.dtypes.to_dict()
        return {
            "table_name": "ACCOUNT",
            "columns": columns,
            "column_types": {col: str(dtype) for col, dtype in column_types.items()}
        }
    except Exception as e:
        st.error(f"Error fetching table info: {str(e)}")
        return None

def get_data_sample():
    try:
        df = execute_sql_query("SELECT * FROM ACCOUNT LIMIT 5")
        return json.loads(json.dumps(df.to_dict('records'), default=json_serial))
    except Exception as e:
        st.error(f"Error fetching data sample: {str(e)}")
        return None

def get_customer_segmentation():
    st.write("Starting customer segmentation analysis...")
    df = execute_sql_query("SELECT * FROM ACCOUNT")
    if df.empty:
        st.error("No data retrieved from the ACCOUNT table")
        return {"error": "No data retrieved from the ACCOUNT table"}
    
    st.write(f"Retrieved {len(df)} rows from ACCOUNT table")
    segmentation = {}
    
    # Customer Segmentation
    for column in ['INDUSTRY', 'TYPE', 'RATING', 'OWNERSHIP', 'ACCOUNTSOURCE']:
        if column in df.columns:
            segmentation[f'{column.lower()}_distribution'] = df[column].value_counts().to_dict()
    
    # Top Customers by Annual Revenue
    if 'ANNUALREVENUE' in df.columns and 'NAME' in df.columns:
        df['ANNUALREVENUE_NUMERIC'] = safe_numeric_convert(df, 'ANNUALREVENUE')
        top_customers = df.nlargest(10, 'ANNUALREVENUE_NUMERIC')[['NAME', 'ANNUALREVENUE_NUMERIC']].to_dict('records')
        segmentation['top_customers_by_revenue'] = top_customers
    
    # Customer Lifetime Value (if available)
    if 'OVERALLLTV__PC' in df.columns and 'NAME' in df.columns:
        df['OVERALLLTV__PC_NUMERIC'] = safe_numeric_convert(df, 'OVERALLLTV__PC')
        top_ltv_customers = df.nlargest(10, 'OVERALLLTV__PC_NUMERIC')[['NAME', 'OVERALLLTV__PC_NUMERIC']].to_dict('records')
        segmentation['top_customers_by_ltv'] = top_ltv_customers
    
    # Lead Source Analysis
    if 'PERSONLEADSOURCE' in df.columns:
        segmentation['lead_source_distribution'] = df['PERSONLEADSOURCE'].value_counts().to_dict()
    
    # Customer Activity
    if 'LASTACTIVITYDATE' in df.columns and 'NAME' in df.columns:
        df['LASTACTIVITYDATE'] = pd.to_datetime(df['LASTACTIVITYDATE'], errors='coerce')
        recent_activity = df.nlargest(10, 'LASTACTIVITYDATE')[['NAME', 'LASTACTIVITYDATE']].to_dict('records')
        segmentation['recent_customer_activity'] = recent_activity
    
    # Review and Rating Analysis
    if 'REVIEWSTARRATING__PC' in df.columns:
        df['REVIEWSTARRATING__PC_NUMERIC'] = safe_numeric_convert(df, 'REVIEWSTARRATING__PC')
        segmentation['rating_distribution'] = df['REVIEWSTARRATING__PC_NUMERIC'].value_counts().to_dict()
    
    # Geographic Analysis
    for column in ['BILLINGCITY', 'BILLINGSTATE', 'BILLINGCOUNTRY']:
        if column in df.columns:
            segmentation[f'{column.lower()}_distribution'] = df[column].value_counts().to_dict()
    
    # Employee Performance
    if 'OWNERID' in df.columns and 'NAME' in df.columns:
        top_employees = df.groupby('OWNERID').size().nlargest(5).to_dict()
        segmentation['top_employees_by_customer_count'] = top_employees
    
    # Customer Engagement (using closed loans as a proxy)
    if 'NO_OF_CLOSED_LOANS__PC' in df.columns and 'NAME' in df.columns:
        df['NO_OF_CLOSED_LOANS__PC_NUMERIC'] = safe_numeric_convert(df, 'NO_OF_CLOSED_LOANS__PC')
        top_engaged = df.nlargest(10, 'NO_OF_CLOSED_LOANS__PC_NUMERIC')[['NAME', 'NO_OF_CLOSED_LOANS__PC_NUMERIC']].to_dict('records')
        segmentation['top_engaged_customers'] = top_engaged
    
    # Data Completeness
    for column in ['PHONE', 'PERSONEMAIL', 'BILLINGSTREET']:
        if column in df.columns:
            completeness = (df[column].notna().sum() / len(df)) * 100
            segmentation[f'{column.lower()}_completeness'] = f"{completeness:.2f}%"
    
    # Additional Analysis
    if 'NUMBEROFEMPLOYEES' in df.columns:
        df['NUMBEROFEMPLOYEES_NUMERIC'] = safe_numeric_convert(df, 'NUMBEROFEMPLOYEES')
        segmentation['company_size_distribution'] = df['NUMBEROFEMPLOYEES_NUMERIC'].describe().to_dict()
    
    if 'PERSONBIRTHDATE' in df.columns:
        df['PERSONBIRTHDATE'] = pd.to_datetime(df['PERSONBIRTHDATE'], errors='coerce')
        df['AGE'] = df['PERSONBIRTHDATE'].apply(lambda x: calculate_age(x) if pd.notnull(x) else np.nan)
        segmentation['age_distribution'] = df['AGE'].describe().to_dict()
    
    st.write("Customer segmentation analysis completed")
    return segmentation

def execute_custom_query(query):
    try:
        df = execute_sql_query(query)
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error executing custom query: {str(e)}")
        return None

def get_data_summary():
    try:
        df = execute_sql_query("SELECT * FROM ACCOUNT")
        summary = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "date_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
        }
        return summary
    except Exception as e:
        st.error(f"Error fetching data summary: {str(e)}")
        return None

def truncate_conversation_history(history, max_tokens=1000):
    total_tokens = sum(len(entry["question"]) + len(entry["answer"]) for entry in history)
    while total_tokens > max_tokens and history:
        removed_entry = history.pop(0)
        total_tokens -= len(removed_entry["question"]) + len(removed_entry["answer"])
    return history

def get_openai_response(question, conversation_history):
    st.write("Generating OpenAI response...")
    try:
        if not st.session_state.openai_api_key:
            st.error("OpenAI API key is missing")
            return "Error: OpenAI API key is missing"

        openai_client = OpenAI(api_key=st.session_state.openai_api_key)
        
        table_info = get_table_info()
        data_summary = get_data_summary()
        
        context = f"Table Information: {json.dumps(table_info)}\n"
        context += f"Data Summary: {json.dumps(data_summary)}\n"
        
        if 'segmentation_data' in st.session_state and st.session_state.segmentation_data:
            segmentation_summary = {k: v for k, v in st.session_state.segmentation_data.items() if k != 'age_distribution'}
            context += f"Segmentation Summary: {json.dumps(segmentation_summary, default=json_serial)}\n"
        else:
            context += "Segmentation data is not available.\n"
        
        truncated_history = truncate_conversation_history(conversation_history)
        
        messages = [
            {"role": "system", "content": f"You are an AI assistant specializing in data analysis and customer information. You have access to a database with customer information. Here's a summary of the data: {context}\n\nAnswer questions based on this information. If you need more specific data to answer a question, you can suggest running a SQL query. If you don't have enough information to answer a question, say so explicitly."}
        ]
        
        # Add truncated conversation history
        for entry in truncated_history:
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})
        
        # Add the current question
        messages.append({"role": "user", "content": question})
        
        st.write("Sending request to OpenAI...")
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        st.write("Received response from OpenAI")
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in get_openai_response: {str(e)}")
        st.error(traceback.format_exc())
        return f"Failed to generate a response: {str(e)}"

# Main interface
st.title("❄️ Ethos Account Data Assistant")

# Snowflake connection parameters
if not st.session_state.snowflake_connected:
    with st.sidebar:
        st.header("Snowflake Connection")
        SNOWFLAKE_ACCOUNT = st.text_input("Account", value="au02318.eu-west-2.aws")
        SNOWFLAKE_USER = st.text_input("Username", value="salesmachinesPOC")
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
    st.header("Ask about Account Data")
    
    if st.button("Perform Customer Segmentation Analysis"):
        with st.spinner("Analyzing customer data..."):
            segmentation_data = get_customer_segmentation()
            if isinstance(segmentation_data, dict) and "error" not in segmentation_data:
                st.session_state.segmentation_data = segmentation_data
                st.success("Customer segmentation analysis completed!")
                st.write("Segmentation data sample:", json.dumps(dict(list(segmentation_data.items())[:5]), indent=2))
            else:
                st.error(segmentation_data.get("error", "An error occurred during segmentation analysis"))
                st.session_state.segmentation_data = None
    
    user_question = st.text_input("Ask a question about the account data:")
    
    if user_question:
        with st.spinner("Generating response..."):
            answer = get_openai_response(user_question, st.session_state.get('conversation_history', []))
            st.subheader("Answer:")
            st.write(answer)
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
            st.session_state.conversation_history.append({"question": user_question, "answer": answer})
    
    # Display conversation history
    if st.checkbox("Show Conversation History"):
        st.subheader("Conversation History")
        if st.session_state.get('conversation_history'):
            for i, entry in enumerate(st.session_state.conversation_history):
                st.write(f"Q{i+1}: {entry['question']}")
                st.write(f"A{i+1}: {entry['answer']}")
                st.write("---")
        else:
            st.write("No conversation history yet.")

else:
    st.warning("Please connect to Snowflake using the sidebar to start analyzing customer data.")

# Display Snowflake Connector version
st.sidebar.write(f"Snowflake Connector Python Version: {snowflake.connector.__version__}")