from typing import List, Dict, Any, Optional
import pandas as pd
import os

# LangChain imports with Gemini instead of Claude
from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI  # Import for Gemini
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_experimental.tools.python.tool import PythonREPLTool
#from langchain.python import PythonREPL
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

class BloodTestAgent:
    """
    A conversational agent specialized in analyzing blood test results using Google's Gemini API.
    """
    
    def __init__(self, api_key=None, model_name="gemini-pro"):
        """
        Initialize the blood test agent.
        
        Args:
            api_key (str, optional): Google API key. If not provided, will look for GOOGLE_API_KEY in environment variables.
            model_name (str, optional): Gemini model to use. Defaults to "gemini-pro".
        """
        # Set API key from argument or environment variable
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize language model with Gemini
        self.llm = ChatGoogleGenerativeAI(
            temperature=0,
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True  # Important for Gemini
        )
        
        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        # Dataframe to store blood test results
        self.df = None
        self.agent = None
        
        # Medical disclaimer
        self.disclaimer = (
            "Note: This information is for educational purposes only and is not intended to replace "
            "professional medical advice. Please consult your healthcare provider for interpretation "
            "of your test results and medical guidance."
        )
    
    def load_dataframe(self, df=None, csv_path=None, excel_path=None):
        """
        Load blood test data from various sources.
        """
        if df is not None and isinstance(df, pd.DataFrame):
            self.df = df
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        elif excel_path:
            self.df = pd.read_excel(excel_path)
        else:
            raise ValueError("Please provide either a DataFrame, CSV path, or Excel path")
        
        # Create the agent with the loaded DataFrame
        self._create_agent()
        
        return self
    
    def _create_agent(self):
        """
        Create the LangChain Pandas DataFrame agent using Gemini.
        """
        if self.df is None:
            raise ValueError("No DataFrame loaded. Call load_dataframe() first.")
        #print("self.df : ", self.df)
        # Create Python REPL tool for dataframe operations
        python_tool = PythonAstREPLTool(
            locals={"df": self.df, "pd": pd},
            description=(
                "A Python shell. Use this to execute python commands. "
                "Input should be a valid python command. "
                "The 'df' variable holds the blood test results DataFrame."
            )
        )
        
        # System message with context about blood tests
        # Note: For Gemini, system messages get converted to human messages
        system_message = (f"""
            You are a helpful assistant specialized in analyzing blood test results. 
            You have access to a dataframe {self.df} containing blood test data. 
            DO NOT create a new dataframe. Use the existing one. 
            When answering questions, always:\
            1. Look up exact values from the dataframe using Python code\n
            2. Include the test value, units, and reference range when discussing a specific test\n
            3. Explain what the test measures and what results might indicate in simple terms\n
            4. Include appropriate medical disclaimers\n\n
            5. DO NOT provide any code samples.  \n\n 
            The dataframe has the following columns:\n
            - test_name: The name of the blood test\n
            - value: The patient's result value\n
            - units: The measurement units\n
            - reference_range: The normal range for this test\n
            - category: The category of the test (optional)\n\n
            Always use the Python tool to query the dataframe rather than making assumptions about the data.      
            """
        )

                
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Set up the agent
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
            }
            | prompt
            | self.llm
            | OpenAIToolsAgentOutputParser()
        )
        
        # Create agent executor
        self.agent = AgentExecutor(
            agent=agent,
            tools=[python_tool],
            verbose=False,
            memory=self.memory,
            handle_parsing_errors=True
        )
    
    def chat(self, query: str) -> str:
        if self.agent is None:
            raise ValueError("Agent not initialized. Load a DataFrame first.")

        try:
            response = self.agent.invoke({"input": query})
            #print("response : ", response)
            
            if isinstance(response, dict):
                if "output" in response:
                    answer = response["output"]
                elif "text" in response:
                    answer = response["text"]
                else:
                    answer = str(response)
            else:
                answer = str(response)

            if any(keyword in query.lower() for keyword in ["test", "level", "result", "value", "range"]):
                if "consult" not in answer.lower():
                    answer += f"\n\n{self.disclaimer}"

            return answer

        except Exception as e:
            return f"I encountered an error processing your question: {str(e)}. Please try rephrasing or asking a different question about your blood test results."

    
    def get_test_names(self) -> List[str]:
        """Get a list of all available test names in the dataset."""
        if self.df is None:
            raise ValueError("No DataFrame loaded.")
        
        return self.df["test_name"].tolist()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.memory.clear()
        
    def reset(self):
        self.chat_history = []