import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os


# Set up the LLM (using Groq or OpenAI's GPT model)
# 🔐 Set your Groq API key
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# 🤖 Initialize Groq LLM (e.g., Mixtral model)
groq_api_key=api_key,
llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct",)

# Chain 1: Extract key ingredients and preferences
chain_one_prompt = ChatPromptTemplate.from_template(
    "Extract the key ingredients and preferences (e.g., savory, quick) from the following input:\n\n{User_Input}"
)
chain_one = LLMChain(llm=llm, prompt=chain_one_prompt, output_key="ingredients_and_preferences")

# Chain 2: Suggest recipe types based on ingredients and preferences
chain_two_prompt = ChatPromptTemplate.from_template(
    "Based on the ingredients and preferences {ingredients_and_preferences}, suggest an recipe."
)
chain_two = LLMChain(llm=llm, prompt=chain_two_prompt, output_key="suggested_recipe_types")

# Chain 3: Suggest cooking time for each recipe
chain_three_prompt = ChatPromptTemplate.from_template(
    "For the following recipes: {suggested_recipe_types}, suggest an approximate cooking time."
)
chain_three = LLMChain(llm=llm, prompt=chain_three_prompt, output_key="cooking_time")

# Chain 4: Provide the final recipe suggestion with all the details
chain_four_prompt = ChatPromptTemplate.from_template(
    "Create a detailed recipe suggestion based on the following information:\n\n"
    "Ingredients: {ingredients_and_preferences}\n"
    "Suggested Recipes: {suggested_recipe_types}\n"
    "Cooking Time: {cooking_time}\n"
    "Provide a clear and friendly message with the recipe do not provide me the cooking time or any other information i need only recipe information."
)
chain_four = LLMChain(llm=llm, prompt=chain_four_prompt, output_key="final_recipe_recommendation")

# Combine all chains into one sequential chain
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["User_Input"],
    output_variables=["final_recipe_recommendation"],
    verbose=False
)

# Streamlit UI
st.title("AI Recipe Suggestion Bot")

# User input for ingredients and preferences
user_input = st.text_area(
    "Describe your ingredients and preferences (e.g., 'I have chicken, tomatoes, and cheese. I want a quick, savory meal.')",
    "")

if st.button("Get Recipe Suggestion"):
    if user_input:
        # Execute the chain and get the final recipe recommendation
        final_recommendation = overall_chain.run(User_Input=user_input)
        # final_meal=final_recommendation['final_recipe_recommendation']
        # Display the final recipe suggestion
        st.subheader("Your Recipe Suggestion:")
        st.write(final_recommendation)
    else:
        st.warning("Please enter your ingredients and preferences.")
