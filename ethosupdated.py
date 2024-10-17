import streamlit as st
from openai import OpenAI
import snowflake.connector
import pandas as pd
import json
import traceback
import time
from collections import deque
import numpy as np
import tiktoken
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state variables
if "snowflake_connected" not in st.session_state:
    st.session_state.snowflake_connected = False
if "snowflake_params" not in st.session_state:
    st.session_state.snowflake_params = {}
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "api_call_times" not in st.session_state:
    st.session_state.api_call_times = deque(maxlen=60)  # Store last 60 API call times

st.title("❄️ Ethos Data Assistant")

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
            logger.info(f"Executing SQL query: {query}")
            cur = conn.cursor()
            cur.execute(query)
            results = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            df = pd.DataFrame(results, columns=columns)
            logger.info(f"Query executed successfully. Returned {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error executing query: {query}")
            logger.error(f"Error message: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()  # Return an empty DataFrame on error
        finally:
            conn.close()

    # Function to get table schema
    def get_table_schema(table_name):
        query = f"DESCRIBE TABLE {table_name}"
        schema = execute_sql_query(query)
        if schema.empty:
            st.warning(f"The {table_name} table schema is empty or the table does not exist.")
        return schema

    # Function to get data summary for a table
    def get_data_summary(table_name, sample_size=1000):
        summary = {}
        df = execute_sql_query(f"SELECT * FROM {table_name} LIMIT {sample_size}")
        if df.empty:
            return {"error": f"The {table_name} table is empty or no data could be retrieved."}
    
        total_rows = execute_sql_query(f"SELECT COUNT(*) FROM {table_name}").iloc[0, 0]
        summary['total_rows'] = int(total_rows)
        summary['columns'] = df.columns.tolist()
    
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if not df[col].isnull().all():
                    summary[f'{col}_avg'] = float(df[col].mean())
                    summary[f'{col}_max'] = float(df[col].max())
                    summary[f'{col}_min'] = float(df[col].min())
                else:
                    summary[f'{col}_avg'] = None
                    summary[f'{col}_max'] = None
                    summary[f'{col}_min'] = None
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                non_null_values = df[col].dropna()
                summary[f'{col}_unique'] = int(non_null_values.nunique())
                if not non_null_values.empty:
                    top_value = non_null_values.value_counts().index[0]
                    summary[f'{col}_top'] = str(top_value)
                else:
                    summary[f'{col}_top'] = None
    
        # Ensure all values are JSON serializable
        for key, value in summary.items():
            if isinstance(value, np.integer):
                summary[key] = int(value)
            elif isinstance(value, np.floating):
                summary[key] = float(value)
            elif isinstance(value, np.ndarray):
                summary[key] = value.tolist()
    
        return summary

    def get_relevant_data(question, schemas, summaries):
        try:
            logger.info("Starting get_relevant_data function")
            logger.info(f"Question: {question}")
            logger.info(f"Schemas: {json.dumps(schemas)}")
            logger.info(f"Summaries: {json.dumps(summaries)}")

        # Combine schemas and summaries for all tables
            all_schema_summary = ""
            all_data_summary = {}
            for table_name in schemas.keys():
                schema_summary = ", ".join([f"{row['name']} ({row['type']})" for _, row in schemas[table_name].iterrows()])
                all_schema_summary += f"{table_name} Schema: {schema_summary}\n"
                all_data_summary[table_name] = {
                    "total_rows": summaries[table_name]["total_rows"],
                    "columns": summaries[table_name]["columns"]
                }
    
        # Truncate summaries if they're too long
            max_length = 3000
            if len(all_schema_summary) > max_length:
                all_schema_summary = all_schema_summary[:max_length] + "..."
                logger.warning(f"Schema summary truncated to {max_length} characters")
            if len(json.dumps(all_data_summary)) > max_length:
                for table in all_data_summary:
                    all_data_summary[table]["columns"] = all_data_summary[table]["columns"][:5]  # Limit to first 5 columns
                all_data_summary["note"] = "Summary truncated due to length"
                logger.warning("Data summary truncated due to length")
    
            openai_client = OpenAI(api_key=st.session_state.openai_api_key)
            messages = [
                {"role": "system", "content": "You are a data analyst. Based on the user's question, determine which columns from the available tables are relevant. Return the relevant columns as a JSON object with table names as keys and lists of column names as values."},
                {"role": "user", "content": f"Question: {question}\nSchemas: {all_schema_summary}\nSummaries: {json.dumps(all_data_summary)}"}
            ]
            logger.info(f"Sending request to OpenAI API: {json.dumps(messages)}")
        
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
        
            logger.info(f"Received response from OpenAI API: {response.choices[0].message.content}")
    
            if not response.choices or not response.choices[0].message.content.strip():
                logger.error("Empty response from OpenAI API")
                raise ValueError("Empty response from OpenAI API")
    
            try:
                relevant_columns = json.loads(response.choices[0].message.content)
                logger.info(f"Parsed relevant columns: {json.dumps(relevant_columns)}")
            except json.JSONDecodeError:
                logger.warning("The AI response was not in the expected JSON format. Using raw response.")
                return {"raw_response": response.choices[0].message.content}
    
            if not isinstance(relevant_columns, dict):
                logger.error(f"The AI response is not in the expected format (dictionary). Received: {type(relevant_columns)}")
                raise ValueError("The AI response is not in the expected format (dictionary)")
    
            relevant_data = {}
            for table, columns in relevant_columns.items():
                if table in summaries:
                    relevant_data[table] = {col: summaries[table].get(f'{col}_avg') or summaries[table].get(f'{col}_unique') or summaries[table].get(f'{col}_top') for col in columns if col in summaries[table]['columns']}
                else:
                    logger.warning(f"Table '{table}' mentioned in AI response is not in the available summaries.")
    
            logger.info(f"Final relevant data: {json.dumps(relevant_data)}")
            return relevant_data

        except Exception as e:
            logger.error(f"Error in get_relevant_data: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to process relevant data: {str(e)}"}

    def check_rate_limit():
        now = time.time()
        st.session_state.api_call_times.append(now)
        if len(st.session_state.api_call_times) == 60:
            oldest_call = st.session_state.api_call_times[0]
            if now - oldest_call < 60:
                wait_time = 60 - (now - oldest_call)
                time.sleep(wait_time)

    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def truncate_context(context, max_tokens=12000):
        """Truncate context to fit within token limit."""
        while num_tokens_from_string(context) > max_tokens:
            context = context[:int(len(context)*0.9)]  # Remove 10% of the context
        return context

    def get_openai_response(question, relevant_data, summaries):
        logger.info("Starting get_openai_response function")
        logger.info(f"Question: {question}")
        logger.info(f"Relevant data: {json.dumps(relevant_data)}")
        logger.info(f"Summaries: {json.dumps(summaries)}")

        if isinstance(relevant_data, str):
            logger.warning(f"Relevant data is a string: {relevant_data}")
            return f"Error in processing relevant data: {relevant_data}"

        try:
            check_rate_limit()  # Implement rate limiting

            openai_client = OpenAI(api_key=st.session_state.openai_api_key)

        # Prepare context
            context = "Available tables and their summaries:\n"
            for table, summary in summaries.items():
                context += f"{table}: {summary['total_rows']} rows, Columns: {', '.join(summary['columns'][:5])}...\n"

            context += "\nRelevant data:\n"
            for table, data in relevant_data.items():
                if isinstance(data, dict):
                    context += f"{table}: {', '.join(data.keys())}\n"
                elif isinstance(data, str):
                    context += f"{table}: {data}\n"
                else:
                    context += f"{table}: Unable to process data of type {type(data)}\n"

        # Truncate context if necessary
            context = truncate_context(context)
            logger.info(f"Prepared context: {context}")

            messages = [
                {"role": "system", "content": f"You are a data analyst. Answer questions about the available data using this context: {context}"}
            ]

        # Add conversation history, but limit it to last 5 exchanges
            history = st.session_state.conversation_history[-10:]
            messages.extend(history)

            messages.append({"role": "user", "content": question})

            logger.info(f"Sending request to OpenAI API: {json.dumps(messages)}")
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            logger.info(f"Received response from OpenAI API: {response.choices[0].message.content}")

            answer = response.choices[0].message.content

            st.session_state.conversation_history.append({"role": "user", "content": question})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})

            if len(st.session_state.conversation_history) > 10:
                st.session_state.conversation_history = st.session_state.conversation_history[-10:]

            return answer

        except Exception as e:
            logger.error(f"Error in get_openai_response: {str(e)}")
            logger.error(traceback.format_exc())
        return "Failed to generate a response"
    
    # Main interface
    st.header("Ask about Your Data")
    user_question = st.text_input("Enter your question about the data:")

    if user_question:
        with st.spinner("Analyzing data..."):
            try:
                logger.info(f"Processing user question: {user_question}")
                tables = ["ACCOUNT", "CONTACT", "OPPORTUNITY", "TASK"]
                schemas = {table: get_table_schema(table) for table in tables}
                summaries = {table: get_data_summary(table) for table in tables}
            
                if all(not schema.empty for schema in schemas.values()):
                    logger.info("Schemas retrieved successfully")
                    st.write("Schemas retrieved successfully")
                    logger.info("Data summaries created")
                    st.write("Data summaries created")
                
                    if all("error" not in summary for summary in summaries.values()):
                        logger.info("Processing relevant data...")
                        st.write("Processing relevant data...")
                        relevant_data = get_relevant_data(user_question, schemas, summaries)
                        logger.info("Relevant data processed")
                        st.write("Relevant data processed")
                        logger.info("Generating OpenAI response...")
                        st.write("Generating OpenAI response...")
                        answer = get_openai_response(user_question, relevant_data, summaries)
                    
                        st.subheader("Answer:")
                        st.write(answer)
                    else:
                        logger.error("Error in data summaries")
                        st.error("Error in data summaries")
                else:
                    logger.error("Failed to retrieve schemas for one or more tables")
                    st.error("Failed to retrieve schemas for one or more tables")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error(traceback.format_exc())

else:
    st.warning("Please connect to Snowflake using the sidebar to start querying data.")

# Display Snowflake Connector version
st.sidebar.write(f"Snowflake Connector Python Version: {snowflake.connector.__version__}")
