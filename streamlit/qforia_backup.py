import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import re
import io
from typing import List, Dict, Any

# App config
st.set_page_config(page_title="Qforia", layout="wide")
st.title("🔍 Qforia: Query Fan-Out Simulator for AI Surfaces")

# Sidebar: API key input and mode selection
st.sidebar.header("Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
mode = st.sidebar.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

# Processing mode selection
processing_mode = st.sidebar.radio("Processing Mode", ["Single Query", "Batch Processing (CSV Upload)"])

# Configure Gemini
if gemini_key:
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
else:
    st.error("Please enter your Gemini API Key to proceed.")
    st.stop()

# Prompt with detailed Chain-of-Thought logic
def QUERY_FANOUT_PROMPT(q, mode):
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"**you must decide on an optimal number of queries to generate.** "
            f"This number must be **at least {min_queries_simple}**. "
            f"For a straightforward query, generating around {min_queries_simple}-{min_queries_simple + 2} queries might be sufficient. "
            f"If the query has a few distinct aspects or common follow-up questions, aim for a slightly higher number, perhaps {min_queries_simple + 3}-{min_queries_simple + 5} queries. "
            f"Provide a brief reasoning for why you chose this specific number of queries. The queries themselves should be tightly scoped and highly relevant."
        )
    else:  # AI Mode (complex)
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"**you must decide on an optimal number of queries to generate.** "
            f"This number must be **at least {min_queries_complex}**. "
            f"For multifaceted queries requiring exploration of various angles, sub-topics, comparisons, or deeper implications, "
            f"you should generate a more comprehensive set, potentially {min_queries_complex + 5}-{min_queries_complex + 10} queries, or even more if the query is exceptionally broad or deep. "
            f"Provide a brief reasoning for why you chose this specific number of queries. The queries should be diverse and in-depth."
        )

    return (
        f"You are simulating Google's AI Mode query fan-out process for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        f"**Your first task is to determine the total number of queries to generate and the reasoning for this number, based on the instructions below:**\n"
        f"{num_queries_instruction}\n\n"
        f"**Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.**\n"
        "Each of the following query transformation types MUST be represented at least once in the generated set, if the total number of queries you decide to generate allows for it (e.g., if you generate 12 queries, try to include all 6 types at least once, and then add more of the relevant types):\n"
        "1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        "The 'reasoning' field for each *individual query* should explain why that specific query was generated in relation to the original query, its type, and the overall user intent.\n"
        "Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        "Return only a valid JSON object. The JSON object should strictly follow this format:\n"
        "{\n"
        "  \"generation_details\": {\n"
        "    \"target_query_count\": 12, // This is an EXAMPLE number; you will DETERMINE the actual number based on your analysis.\n"
        "    \"reasoning_for_count\": \"The user query was moderately complex, so I chose to generate slightly more than the minimum for a simple overview to cover key aspects like X, Y, and Z.\" // This is an EXAMPLE reasoning; provide your own.\n"
        "  },\n"
        "  \"expanded_queries\": [\n"
        "    // Array of query objects. The length of this array MUST match your 'target_query_count'.\n"
        "    {\n"
        "      \"query\": \"Example query 1...\",\n"
        "      \"type\": \"reformulation\",\n"
        "      \"user_intent\": \"Example intent...\",\n"
        "      \"reasoning\": \"Example reasoning for this specific query...\"\n"
        "    },\n"
        "    // ... more query objects ...\n"
        "  ]\n"
        "}"
    )

# Fan-out generation function
def generate_fanout(query, mode):
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip()
        
        # Clean potential markdown code block fences
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        data = json.loads(json_text)
        generation_details = data.get("generation_details", {})
        expanded_queries = data.get("expanded_queries", [])

        # Store details for display
        st.session_state.generation_details = generation_details

        return expanded_queries
    except json.JSONDecodeError as e:
        st.error(f"🔴 Failed to parse Gemini response as JSON: {e}")
        st.text("Raw response that caused error:")
        st.text(json_text if 'json_text' in locals() else "N/A (error before json_text assignment)")
        st.session_state.generation_details = None
        return None
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred during generation: {e}")
        if hasattr(response, 'text'):
             st.text("Raw response content (if available):")
             st.text(response.text)
        st.session_state.generation_details = None
        return None

# Initialize session state for generation_details if not present
if 'generation_details' not in st.session_state:
    st.session_state.generation_details = None

# Batch processing functions
def validate_csv_upload(uploaded_file) -> tuple[bool, str, pd.DataFrame]:
    """Validate uploaded CSV file and return validation status, message, and dataframe."""
    if uploaded_file is None:
        return False, "No file uploaded", None
    
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if dataframe is empty
        if df.empty:
            return False, "CSV file is empty", None
        
        # Check for required columns (at least one column should contain queries)
        if len(df.columns) == 0:
            return False, "CSV file has no columns", None
        
        # If there's only one column, assume it contains queries
        if len(df.columns) == 1:
            query_column = df.columns[0]
        else:
            # If multiple columns, look for a column that might contain queries
            # Common column names for queries
            query_column_names = ['query', 'queries', 'keyword', 'keywords', 'text', 'input', 'prompt']
            query_column = None
            
            for col in df.columns:
                if col.lower() in query_column_names:
                    query_column = col
                    break
            
            # If no matching column found, use the first column
            if query_column is None:
                query_column = df.columns[0]
        
        # Check if the selected column has any non-empty values
        if df[query_column].isna().all() or (df[query_column] == '').all():
            return False, f"Column '{query_column}' contains no valid queries", None
        
        # Remove empty rows
        df_clean = df.dropna(subset=[query_column])
        df_clean = df_clean[df_clean[query_column] != '']
        
        if df_clean.empty:
            return False, "No valid queries found after removing empty rows", None
        
        return True, f"Valid CSV with {len(df_clean)} queries in column '{query_column}'", df_clean
    
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}", None

def process_batch_queries(queries: List[str], mode: str) -> List[Dict[str, Any]]:
    """Process multiple queries and return combined results."""
    all_results = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, query in enumerate(queries):
        status_text.text(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
        
        try:
            # Generate fan-out for this query
            expanded_queries = generate_fanout(query, mode)
            
            if expanded_queries:
                # Add metadata to each result
                for j, result in enumerate(expanded_queries):
                    result_with_metadata = result.copy()
                    result_with_metadata['original_query'] = query
                    result_with_metadata['query_index'] = i + 1
                    result_with_metadata['batch_processing'] = True
                    all_results.append(result_with_metadata)
            
            # Update progress
            progress = (i + 1) / len(queries)
            progress_bar.progress(progress)
            
        except Exception as e:
            st.error(f"Error processing query {i+1}: {str(e)}")
            # Continue with next query
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return all_results

# Generate and display results
if st.sidebar.button("Run Fan-Out 🚀"):
    # Clear previous details
    st.session_state.generation_details = None
    
    if processing_mode == "Single Query":
        if not mode:
            st.warning("⚠️ Please select a Search Mode.")
        elif not user_query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("🤖 Generating query fan-out using Gemini... This may take a moment..."):
                results = generate_fanout(user_query, mode)

            if results: # Check if results is not None and not empty
                st.success("✅ Query fan-out complete!")

                # Display the reasoning for the count if available
                if st.session_state.generation_details:
                    details = st.session_state.generation_details
                    generated_count = len(results)
                    target_count_model = details.get('target_query_count', 'N/A')
                    reasoning_model = details.get('reasoning_for_count', 'Not provided by model.')

                    st.markdown("---")
                    st.subheader("🧠 Model's Query Generation Plan")
                    st.markdown(f"🔹 **Target Number of Queries Decided by Model:** `{target_count_model}`")
                    st.markdown(f"🔹 **Model's Reasoning for This Number:** _{reasoning_model}_")
                    st.markdown(f"🔹 **Actual Number of Queries Generated:** `{generated_count}`")
                    st.markdown("---")
                    
                    if isinstance(target_count_model, int) and target_count_model != generated_count:
                        st.warning(f"⚠️ Note: Model aimed to generate {target_count_model} queries but actually produced {generated_count}.")
                else:
                     st.info("ℹ️ Generation details (target count, reasoning) were not available from the model's response.")


                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, height=(min(len(df), 20) + 1) * 35 + 3) # Dynamic height

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="qforia_output.csv", mime="text/csv")
            
            elif results is None: # Error occurred in generate_fanout
                # Error message is already displayed by generate_fanout
                pass
            else: # Handle empty results list (empty list, not None)
                st.warning("⚠️ No queries were generated. The model returned an empty list, or there was an issue.")
    elif processing_mode == "Batch Processing (CSV Upload)":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            validation_status, message, df_clean = validate_csv_upload(uploaded_file)
            
            if validation_status:
                st.success(f"✅ CSV file validated: {message}")
                st.info(f"Found {len(df_clean)} valid queries in the '{df_clean.columns[0]}' column.")
                
                with st.spinner("🤖 Processing batch queries using Gemini... This may take a moment..."):
                    results = process_batch_queries(df_clean[df_clean.columns[0]].tolist(), mode)
                
                if results:
                    st.success("✅ Batch processing complete!")
                    
                    # Combine results into a single DataFrame
                    combined_df = pd.DataFrame(results)
                    
                    st.dataframe(combined_df, use_container_width=True, height=(min(len(combined_df), 20) + 1) * 35 + 3) # Dynamic height
                    
                    # Download combined results
                    csv = combined_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Combined CSV", data=csv, file_name="qforia_batch_output.csv", mime="text/csv")
                    
                    # Display individual query results
                    st.subheader("Individual Query Results (for a few examples)")
                    st.dataframe(combined_df[combined_df['batch_processing'] == True].head(5), use_container_width=True, height=(min(5, len(combined_df[combined_df['batch_processing'] == True])) + 1) * 35 + 3)
                else:
                    st.warning("⚠️ No queries were generated for the batch. The model returned an empty list, or there was an issue.")
            else:
                st.error(f"❌ CSV file validation failed: {message}")
        else:
            st.warning("⚠️ Please upload a CSV file for batch processing.")