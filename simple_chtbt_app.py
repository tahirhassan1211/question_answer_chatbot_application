from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq  # Make sure this is available and installed correctly

# Load environment variables
load_dotenv()

# --- Step 1: Load the LLaMA model from Groq ---
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7  # Change to the actual model name if needed
)

# --- Step 2: Define the Prompt Template ---
prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful information assistant. Answer the user's question clearly and concisely.\n Question:{question} \n Answer:"
)

# 3. Define output parser
parser = StrOutputParser()

# 4. Compose runnable chain (prompt → LLM → parse)
chain: Runnable = prompt | llm | parser

# --- Step 3: Create a Chain ---
#chain = LLMChain(
#    llm=llm,
#   prompt=prompt_template,
#    output_parser=StrOutputParser()
#)

# --- Step 4: Chat Interface ---
def chatbot():
    print("🤖 InfoBot is ready! Ask any question. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break

        response = chain.invoke({"question": user_input})
        print(f"\nBot: {response.strip()}\n")

if __name__ == "__main__":
    chatbot()
