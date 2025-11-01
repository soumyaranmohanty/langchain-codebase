from dotenv import load_dotenv

#Import prompt templates
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI


from langchain import hub 
from langchain.agents import AgentExecutor 
from langchain.agents.react.agent import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun()
from langchain.tools import Tool

# Load environment variables from a .env file

load_dotenv()



def main():
    # ✅ Get the official ReAct prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    #Defile tools
    tools = [DuckDuckGoSearchRun(), WikipediaQueryRun()]

    """
    Create a ReAct agent using the prompt and llm
    
    """

    agent = create_react_agent(llm, tools=tools, prompt=react_prompt)

    # Create an AgentExecutor to run the agent
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Invoke the agent with input
    response = agent_executor.invoke(input={"input" : "Search for 3 jobs in linkedin for 'data scientist' in 'New York' and list their details."})

    # Print the response
    print(response) 


if __name__ == '__main__':
    main()
