# Import necessary modules from LangChain and other libraries
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the large language model (LLM) using GoogleGenerativeAI with the specified model
# The API key is retrieved from environment variables for authentication
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Try to load the document "national_ai_policy.txt" using the TextLoader class
# Catch and print any exceptions that occur during file loading
try:
    loader = TextLoader("national_ai_policy.txt")
except Exception as e:
    print("Error while loading file=", e)

# Create embeddings using GoogleGenerativeAIEmbeddings
# This is used to transform text data into vector format for further processing
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize a text splitter to manage text chunk size and overlap
# This helps to ensure that text input stays within token limits for the LLM
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Create an index creator using the specified embedding model and text splitter
# This index is used to efficiently search and retrieve information from the loaded text
index_creator = VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter
)

# Build the index using the document loader
index = index_creator.from_loaders([loader])

# Enter an infinite loop to continuously query the index using the LLM
# The user inputs a query, and the response is generated and printed
while True:
    human_message = input("How can I help you today? ")
    response = index.query(human_message, llm=llm)
    print(response)
