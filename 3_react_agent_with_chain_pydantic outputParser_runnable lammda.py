
from dotenv import load_dotenv
#Import prompt templates
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor 
from langchain.agents.react.agent import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun

from langchain_core.runnables import RunnableLambda
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse


# Load environment variables from a .env file

load_dotenv()



def main():
    # ✅ Get the official ReAct prompt from LangChain Hub
    #react_prompt = hub.pull("hwchase17/react")

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    #Defile tools
    tools = [DuckDuckGoSearchRun(), WikipediaQueryRun()]

    output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS, input_variables=["input", "agent_scratchpad", "tool_names"]
    ).partial(format_instructions=output_parser.get_format_instructions())

    """
    Create a ReAct agent using the prompt and llm
    
    """

    agent = create_react_agent(llm, tools=tools, prompt=react_prompt_with_format_instructions)

    # Create an AgentExecutor to run the agent
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    extract_output = RunnableLambda(lambda x: ["output"])

    parse_output = RunnableLambda(lambda x: output_parser.parse (x))

    chain = agent_executor | extract_output | parse_output

    # Invoke the agent with input
    response = chain.invoke(input={"input" : "Search for 3 jobs in linkedin for 'data scientist' in 'New York' and list their details."})

    # Print the response
    print(response) 


if __name__ == '__main__':
    main()



