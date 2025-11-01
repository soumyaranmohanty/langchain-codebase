from dotenv import load_dotenv

import pandas as pd

#Import prompt templates
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

# Load environment variables from a .env file

load_dotenv()



def main():
    # Define a prompt template
    template = "What is a good name for a company that makes {product}?"


    prompt = PromptTemplate(
        input_variables=["product"],
        template=template,
    )

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    

    """
    Manual method of formating the prompt and invoking the llm
    
    """
    
    # Format the prompt with a specific product
    #formatted_prompt = prompt.format(product="colorful socks")

    # Get the response from the model
    #response = chat(formatted_prompt)



    """
    Chain method of invoking the prompt and llm
    
    """

    #create chain with prompt and model
    chain = prompt | llm
    #invoke chain with input
    response = chain.invoke({"product": "colorful socks"})

    # Print the response
    print(response)

if __name__ == '__main__':
    main()