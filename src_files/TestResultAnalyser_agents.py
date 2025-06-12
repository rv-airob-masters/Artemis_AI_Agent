# Main functionality:
# 1. Extracts medical test results from PDFs and images using OCR and Azure Document Intelligence
# 2. Analyzes test results against reference ranges from a SQLite database
# 3. Identifies abnormal results (high/low) and provides explanations
# 4. Offers a chat interface with an AI agent to answer questions about results

# Key components:
# - Data extraction: PDF parsing, OCR, Azure Document Intelligence
# - Data processing: Reference range matching, abnormality detection
# - AI integration: Google Gemini for analysis, BloodTestAgent for chat
# - UI: Streamlit interface with tabs for results and chat


import streamlit as st
import pdfplumber
import pytesseract
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from langchain_community.chat_models.anthropic import ChatAnthropic
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool, StructuredTool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
import json
import re
import base64
from PyPDF2 import PdfReader
from difflib import get_close_matches
import sqlite3
from dotenv import load_dotenv
from anthropic import Anthropic
import os
from typing import Optional, Type, Dict, Any, ClassVar, Mapping

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool

from blood_test_agent import BloodTestAgent
#from extract_image_text import extract_structured_test_results
from ocr_library import extract_data_from_image
from azure_document_intelligence import analyze_layout_from_uploaded_file
from medical_analyzer import process_dataframe
# Load environment variables
load_dotenv()



google_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=google_api_key)
#model_name = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0)
model_name = "gemini-1.5-pro"


st.markdown(
    """
    <style>
    /* Main app and all text inside it */
    .stApp {
        color: black;
    }

    .stButton > button {
        color: white !important;      /* Text colour */
        background-color: #4CAF50;    /* Optional: green background */
        border: none;                 /* Optional: no border */
    }

    .stButton > button:hover {
        background-color: #45a049;    /* Optional: hover effect */
    }

    /* Headers, paragraphs, spans, and labels */
    h1, h2, h3, h4, h5, h6,
    p, span, div, label, li, a {
        color: black !important;
    }

    /* Sidebar elements */
    [data-testid="stSidebar"] * {
        color: White !important;
    }

    /* Widget labels */
    .css-1cpxqw2, .css-1cpxqw2 label, .css-1cpxqw2 span {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def set_full_bg(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0);  /* transparent sidebar */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_full_bg("../assets/background.png")



# Function to load reference ranges from SQLite database
@st.cache_data
def load_reference_ranges_from_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT test_name, age_group, gender, range
        FROM reference_ranges
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        st.error(f"Error reading database: {e}")
        return pd.DataFrame()


def flatten_reference_ranges_from_df(df):
    flat_ranges = {}

    for _, row in df.iterrows():
        test_name = row["test_name"]
        age_group = row["age_group"]
        gender = row["gender"]
        ref_range = row["range"]

        key = f"{test_name.lower().strip()}_{gender}_{age_group}"
        flat_ranges[key] = {
            "Reference Range": ref_range,
            "Original Name": test_name,
            "Gender": gender,
            "Age Group": age_group
        }
    return flat_ranges



def reference_ranges_to_df(flat_ranges):
    data = []
    for test_name, values in flat_ranges.items():
        data.append({
            "Test Name": values.get("Original Name", test_name.title()),
            "Gender": values.get("Gender", ""),
            "Age Group": values.get("Age Group", ""),
            "Reference Range": values.get("Reference Range", "")

        })
    return pd.DataFrame(data).sort_values("Test Name")


def find_matching_reference(test_name, flat_ranges, threshold=0.6):
    test_name_lower = test_name.lower().strip()

    if test_name_lower in flat_ranges:
        return flat_ranges[test_name_lower]

    for key in flat_ranges:
        if key in test_name_lower or test_name_lower in key:
            return flat_ranges[key]

    test_keys = list(flat_ranges.keys())
    matches = get_close_matches(test_name_lower, test_keys, n=1, cutoff=threshold)
    if matches:
        return flat_ranges[matches[0]]

    return None

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text if text else "No text found."

# def extract_text_from_image(image_file):
    # image = Image.open(image_file)
    # text = pytesseract.image_to_string(image)

    # return text if text else "No text found."

def extract_medical_test_results(text, reference_ranges=None):
    genai.configure(api_key=google_api_key)
    
    reference_info = ""
    if reference_ranges:
        # Send a subset of reference ranges to avoid token limits
        sample_ranges = dict(list(reference_ranges.items())[:20]) 
        reference_info = "Use these reference ranges as examples (extract all test results even if not in this list):\n"
        reference_info += json.dumps(sample_ranges, indent=2)
    
    prompt = f"""
    Extract only medical test results from the following text. Ignore any unrelated content.
    Present the extracted results in a structured JSON format as a list of dictionaries.
    Each dictionary should have fields: "Test Name", "Value", "Reference Range".
    For the "Value" field, extract only the numeric value without any units.
    Leave the "Reference Range" field empty if not provided in the text.
    
    Your response must be valid JSON only, with no explanatory text before or after.
    {reference_info}
    
    Text:
    {text}
    """
    
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    raw_response = response.text
    
    
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\[\s*{.*}\s*\]', raw_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response)
            if code_block_match:
                return json.loads(code_block_match.group(1))

            return {"error": "Could not parse JSON response", "raw_response": raw_response}
        except Exception as e:
            return {"error": f"Error extracting JSON: {str(e)}", "raw_response": raw_response}

def format_results_as_dataframe(results, reference_ranges=None):
    try:
        if isinstance(results, dict) and "error" in results:
            st.error(f"Error parsing results: {results.get('error')}")
            st.code(results.get('raw_response', ''))
            return pd.DataFrame(columns=["Test Name", "Value", "Reference Range", "Status"])
            
        if not isinstance(results, list) or len(results) == 0:
            return pd.DataFrame(columns=["Test Name", "Value", "Reference Range", "Status"])

        # for item in results:
            # if "Unit" not in item:
                # item["Unit"] = ""
                
        df = pd.DataFrame(results)
        
        if reference_ranges:
            for index, row in df.iterrows():
                test_name = row["Test Name"]

                if "Reference Range" not in row or pd.isna(row["Reference Range"]) or row["Reference Range"] == "":
                    ref_data = find_matching_reference(test_name, reference_ranges)

                    if ref_data:
                        df.at[index, "Reference Range"] = ref_data["Reference Range"]
                        if (pd.isna(row.get("Unit", "")) or row.get("Unit", "") == "") and ref_data["Unit"]:
                            df.at[index, "Unit"] = ref_data["Unit"]

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        def check_abnormality(row):
            if pd.isna(row["Value"]) or not row["Reference Range"] or pd.isna(row["Reference Range"]):
                return "Unknown"

            ref_range = str(row["Reference Range"])
            value = row["Value"]

            range_match = re.search(r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)", ref_range)
            if range_match:
                low, high = float(range_match.group(1)), float(range_match.group(2))
                if value < low:
                    return "Low"
                elif value > high:
                    return "High"
                else:
                    return "Normal"

            less_than_match = re.search(r"<\s*(\d+\.?\d*)", ref_range)
            if less_than_match:
                threshold = float(less_than_match.group(1))
                return "Normal" if value < threshold else "High"

            greater_than_match = re.search(r">\s*(\d+\.?\d*)", ref_range)
            if greater_than_match:
                threshold = float(greater_than_match.group(1))
                return "Normal" if value > threshold else "Low"

            less_equal_match = re.search(r"(?:≤|<=)\s*(\d+\.?\d*)", ref_range)
            if less_equal_match:
                threshold = float(less_equal_match.group(1))
                return "Normal" if value <= threshold else "High"

            greater_equal_match = re.search(r"(?:≥|>=)\s*(\d+\.?\d*)", ref_range)
            if greater_equal_match:
                threshold = float(greater_equal_match.group(1))
                return "Normal" if value >= threshold else "Low"

            return "Unknown"

        df["Status"] = df.apply(check_abnormality, axis=1)
        return df

    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        return pd.DataFrame(columns=["Test Name", "Value", "Reference Range", "Status"])

def query_medical_explanations(test_name, status):

    genai.configure(api_key=google_api_key)
    prompt = f"""
    Explain the possible causes and remedies for an abnormal {test_name} test result classified as '{status}'.
    Provide a clear, concise medical explanation.

    Constraints
    - Maintain medical accuracy and avoid alarmist language
    - Clarify when multiple interpretations are possible
    - Emphasize that this analysis is not a diagnosis
    - Indicate when immediate medical attention is warranted
    """
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    
    return response.text

def scrape_nhs(test_name):
    search_url = f"https://www.nhs.uk/search/?query={test_name.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(search_url, headers=headers, timeout=5)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all("a", href=True)
            for link in links:
                if "/conditions/" in link["href"]:
                    return f"https://www.nhs.uk{link['href']}"
    except requests.RequestException as e:
        st.warning(f"Could not fetch NHS information: {e}")
        
    return "No NHS page found."

# New tool for the agent to get test information
class TestResultLookupTool(BaseTool):
    name: ClassVar[str] = "test_result_lookup"
    description: ClassVar[str] = "Use this tool to look up specific test results by providing the test name."
    results_df: Optional[pd.DataFrame] = None
    
    def _run(self, test_name: str) -> str:
        if self.results_df is None or self.results_df.empty:
            return "No test results available."
        
        # Case-insensitive search with partial matches
        matches = self.results_df[self.results_df["Test Name"].str.lower().str.contains(test_name.lower())]
        
        if matches.empty:
            return f"No test results found for '{test_name}'."
        
        # Format the result as a readable string
        results = []
        for _, row in matches.iterrows():
            unit = row.get("Unit", "")
            result_str = (f"Test: {row['Test Name']}\n"
                         f"Value: {row['Value']} {unit}\n"
                         f"Reference Range: {row['Reference Range']} {unit}\n"
                         f"Status: {row['Status']}")
            results.append(result_str)
        
        return "\n\n".join(results)

    def _arun(self, test_name: str) -> str:
        """Use the tool asynchronously."""
        return self._run(test_name)

# New tool for medical explanations
class MedicalExplanationTool(BaseTool):
    name: ClassVar[str] = "medical_explanation"
    description: ClassVar[str] = "Get an explanation for a specific medical test and its abnormal status (Low or High)."
    
    def _run(self, query: str) -> str:
        try:
            # Parse query - expect format like "test_name:status"
            parts = query.split(":")
            if len(parts) != 2:
                return "Please provide both test name and status in format 'test_name:status'"
            
            test_name = parts[0].strip()
            status = parts[1].strip()
            
            if status not in ["Low", "High"]:
                return "Status must be either 'Low' or 'High'"
                
            explanation = query_medical_explanations(test_name, status)
            return explanation
            
        except Exception as e:
            return f"Error getting medical explanation: {str(e)}"

    def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        return self._run(query)

def query_dataframe(query: str) -> str:
    return str(results_df.query(query))
# New function to create the Langchain agent


# Function to display chat interface
def display_chat_interface(agent_executor):
    st.subheader("Chat with AI about your Test Results")
    st.write("Ask questions about your test results and get detailed information from our AI assistant.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I can help explain your test results. What would you like to know?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask about your test results...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.chat(prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Show dataframe with wrapped text using st.dataframe + styling
def show_wrapped_dataframe(df):
    if df is not None and not df.empty:
        styled_df = df.style.set_properties(**{
            'white-space': 'pre-wrap',
            'word-wrap': 'break-word'
        })
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("No data available to display.")



# Main app
def main():
    # Initialize session state for analysis complete flag
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    if 'medical_results_df' not in st.session_state:
        st.session_state.medical_results_df = None
    
    if 'agent_created' not in st.session_state:
        st.session_state.agent_created = False
    
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
        
    db_path = "db/med_reference_ranges.db"
    reference_ranges_df = load_reference_ranges_from_db(db_path)

    if not reference_ranges_df.empty:
        flat_reference_ranges = flatten_reference_ranges_from_df(reference_ranges_df)
        #print("flat_reference_ranges : \n", flat_reference_ranges)

    
    with st.sidebar:
        st.title("Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

        
        st.markdown("###")
        if st.button("🔄 Reset Chat", help="Clear chat history with the AI"):
            # Reset agent if exists
            if st.session_state.get("agent"):
                st.session_state.agent.reset()

            # Only clear chat-related state, not file or result data
            keys_to_clear = [
                key for key in st.session_state.keys()
                if key.startswith("chat_") or "message" in key.lower() or key.startswith("nhs_link_") or key.startswith("explanation_")
            ]
            for key in keys_to_clear:
                del st.session_state[key]

            st.success("Chat history reset.")
            st.rerun()       
        
        
        st.subheader("Reference Ranges")
        if not reference_ranges_df.empty:
            st.success(f"✅ Loaded {len(flat_reference_ranges)} haematology reference ranges")
            if st.checkbox("Show available reference ranges"):
                ranges_df = reference_ranges_to_df(flat_reference_ranges)
                st.dataframe(ranges_df, hide_index=True, use_container_width=True)

                search_term = st.text_input("Search for a specific test", "")
                if search_term:
                    filtered_df = ranges_df[ranges_df["Test Name"].str.lower().str.contains(search_term.lower())]
                    if not filtered_df.empty:
                        st.subheader(f"Search results for '{search_term}'")
                        st.dataframe(filtered_df, hide_index=True, use_container_width=True)
                    else:
                        st.info(f"No tests found matching '{search_term}'")
        else:
            st.warning("⚠️ No haematology reference ranges loaded")

    azure_endpoint = os.getenv("AZURE_DOCU_URL")
    azure_key = os.getenv("AZURE_DOCU_KEY")

    if uploaded_file:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
            reader = PdfReader(uploaded_file)
            num_pages = len(reader.pages)
            st.sidebar.info(f"PDF document with {num_pages} pages")

            uploaded_file.seek(0)

            base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'

            st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

        elif file_type.startswith("image"):
            image = Image.open(uploaded_file)
            #extracted_text = extract_structured_test_results(uploaded_file)
            #extracted_text = extract_data_from_image(uploaded_file)
            extracted_text, _ = analyze_layout_from_uploaded_file(uploaded_file, azure_endpoint, azure_key)
            #print("Extracted text:")
            #print(extracted_text)
            #extracted_text = extract_text_from_image(uploaded_file)
            st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            extracted_text = "Unsupported file type."

    # Main content
    st.title("Medical Test Result Analyzer")
    
    # Show tabs after analysis is complete
    if st.session_state.analysis_complete:
        tab1, tab2 = st.tabs(["Test Results Analysis", "Chat with AI"])
        
        with tab1:
            st.subheader("Medical Test Results")
            if st.session_state.medical_results_df is not None and not st.session_state.medical_results_df.empty:
                reference_stats = {
                    "Total tests found": len(st.session_state.medical_results_df),
                    "Tests with reference ranges": st.session_state.medical_results_df["Reference Range"].notna().sum(),
                    "Tests with unknown status": (st.session_state.medical_results_df["Status"] == "Unknown").sum()
                }

                st.dataframe(st.session_state.medical_results_df, use_container_width=True)

                cols = st.columns(3)
                for i, (stat, value) in enumerate(reference_stats.items()):
                    cols[i].metric(stat, value)

                abnormal_tests = st.session_state.medical_results_df[st.session_state.medical_results_df["Status"].isin(["Low", "High"])]
                if not abnormal_tests.empty:
                    st.subheader("Abnormal Test Results & Explanations")
                    if st.session_state.analysis_results is None: 
                        try:
                            st.session_state.analysis_results = process_dataframe(abnormal_tests)
                        except ValueError as e:
                            st.error(f"Configuration error: {str(e)}")
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                    #st.dataframe(analysis_results)
                    
                    if st.session_state.analysis_results is not None: 
                        formatted_output = st.session_state.analysis_results['output'].replace("<br>", "\n")
                        st.markdown(formatted_output)


                else:
                    st.success("All test results are within normal ranges.")
            else:
                st.write("No valid medical test results found.")
        
        with tab2:
            # Create agent if not already created
            if not st.session_state.agent_created:
                try:
                    if google_api_key:
                        print("Creating agent")
                        #print("Medical result df : \n", st.session_state.medical_results_df)

                        st.session_state.agent = BloodTestAgent(api_key=google_api_key, model_name=model_name)
                        print("Initilized agent")
                        st.session_state.agent.load_dataframe(df=st.session_state.medical_results_df)
                        print("Agent creation done")
                        st.session_state.agent_created = True
                    else:
                        st.error("GOOGLE_API_KEY not found in environment variables.")
                except Exception as e:
                    st.error(f"Error creating AI agent: {str(e)}")
            
            # Display chat interface if agent is created
            if st.session_state.agent_created:
                display_chat_interface(st.session_state.agent)
            else:
                if not google_api_key:
                    st.error("GOOGLE_API_KEY API key not found. Please check your .env file and ensure GOOGLE_API_KEY is set.")
                else:
                    st.error("Unable to create AI agent. Please check the error message above.")
    else:
        # Initial upload and analysis workflow
        st.write("Upload a medical report (PDF or Image) from the sidebar, and we will extract and analyze the test results.")

        if uploaded_file:
            st.subheader("Extracted Text")
            st.text_area(" ", extracted_text, height=200)

            if st.button("Continue to Analysis"):
                with st.spinner("Analysing medical test results..."):
                    try:
                        medical_results = extract_medical_test_results(extracted_text, flat_reference_ranges)
                        
                        if isinstance(medical_results, dict) and "error" in medical_results:
                            st.error(f"Error extracting results: {medical_results.get('error')}")
                            st.code(medical_results.get('raw_response', ''))
                        else:
                            medical_results_df = format_results_as_dataframe(medical_results, flat_reference_ranges)
                            
                            # Store results in session state
                            st.session_state.medical_results_df = medical_results_df
                            st.session_state.analysis_complete = True
                            
                            # Rerun to display the tabs
                            st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
st.divider()
st.caption("""
**Medical Disclaimer**: This application provides information for educational purposes only. 
It is not intended as a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of your physician or other qualified health provider with any questions 
you may have regarding a medical condition or test results.
""")
st.divider()

if __name__ == "__main__":
    main()
