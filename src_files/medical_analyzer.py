# medical_analysis.py
#from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import tool
from langchain_core.prompts import MessagesPlaceholder
import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from typing import Dict, List


class TestDataInput(BaseModel):
    test_data: Dict[str, str] 

class TestResult(BaseModel):
    test_name: str
    status: str
    probable_causes: List[str]
    remedies: List[str]
    nhs_site_address: str
    medical_disclaimer: str

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True
    
    
# Define helper functions first
def parse_explanation(text: str):
    """Extract structured data from LLM response"""
    causes = []
    remedies = []
    nhs_link = "https://www.nhs.uk/conditions/"

    # Extract Causes
    if "Probable Causes" in text:
        causes_section = text.split("Probable Causes")[1].split("Remedies")[0]
        causes = [line.strip() for line in causes_section.split("- ") if line.strip()]
    
    # Extract Remedies
    if "Remedies" in text:
        remedies_section = text.split("Remedies")[1].split("NHS Link")[0]
        remedies = [line.strip() for line in remedies_section.split("- ") if line.strip()]
    
    # Extract NHS Link
    if "NHS Link" in text:
        nhs_section = text.split("NHS Link")[1]
        nhs_link = nhs_section.split(":")[1].strip() if ":" in nhs_section else nhs_link

    return {
        "Probable Causes": causes[:3] or ["See NHS link"],
        "Remedies": remedies[:3] or ["Medical consultation required"],
        "NHS Link": nhs_link
    }

def extract_section(text: str, section_name: str):
    """Helper function to extract sections from LLM output"""
    return text.split(section_name)[1].split("\n\n")[0].strip()



# Define tool
@tool(args_schema=TestDataInput)
def analyze_test_results(test_data: dict) -> list:
    """Tool that returns a list of analysis dicts"""
    return [
        parse_explanation(query_medical_explanations(name, status))
        for name, status in test_data.items()
    ]

# Define processing functions
def initialize_agent():
    #llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    llm = ChatMistralAI(
        model="mistral-large-2411",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")  
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical analysis assistant. Process test results with:
        - Clinical accuracy
        - Clear risk stratification
        - Appropriate referrals
        - Patient-friendly language"""),
        ("user", """Analyze these test results: {input}
        
        Required Output Format for Each Test:
        - Test name: [name]
        - Status: [status]
        - Probable causes: [brief explanation under 50 words]
        - Remedies: [brief explanations under 50 words]
        
        After all the results, provide  Medical disclaimer: [standard text] with a link to NHS website for conditions.

        Always provide the above output in a pandas dataframe layout.
        
        """),
        MessagesPlaceholder("agent_scratchpad") 
    ])
    tools = [analyze_test_results]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def process_dataframe(df):
    agent_executor = initialize_agent()
    print("Agent initialized")
    #print("Df received: ",df)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    required_columns = {'test_name', 'status'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")
    
    test_data = dict(zip(df['test_name'], df['status']))
    #print("test_data : \n", test_data)
    try:
        result = agent_executor.invoke(
            {"input": dict(test_data)},
            handle_parsing_errors=True  # Critical for tool errors
        )
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        return None
    #print("result : \n ", result)
    

    return result

def convert_to_dataframe(converted_dict):
    data = []

    for test_name, info in converted_dict.items():
        causes = " ".join(info.get("Probable Causes", []))
        remedies = " ".join(info.get("Remedies", []))
        nhs_link = info.get("NHS Link", "")
        data.append({
            "Test Name": test_name,
            "Probable Causes": causes,
            "Remedies": remedies,
            "NHS Link": nhs_link
        })

    df = pd.DataFrame(data)
    return df

 
# Note: query_medical_explanations() needs implementation
def query_medical_explanations(test_name: str, status: str) -> str:
    # llm = ChatAnthropic(
        # model="claude-3-opus-20240229",
        # temperature=0,
        # anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    # )
    llm = ChatMistralAI(
        model="mistral-large-2411",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")  
    )
    prompt = f"""
    Analyze {test_name} ({status}) and provide structured output:

    **Probable Causes**  
    - [List 3 medical causes with brief explanations]  
    
    **Remedies**  
    - Dietary: [Specific food recommendations]  
    - Lifestyle: [Exercise/sleep/stress management]  
    - Medical: [Treatment options]  
    
    
    Format strictly as shown above with bold headers and bullet points.
    """

    return llm.invoke(prompt).content
