from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the LLM
llm = Ollama(model="llama3.2")

# Load and split document
loader = PyPDFLoader("the_lightning_thief.pdf")
docs = loader.load_and_split()

# Embed and store in vector DB
embedding = OllamaEmbeddings()
db = FAISS.from_documents(docs, embedding)

# Create a retriever and a QA chain
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    # Get user input
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Get the answer from the QA chain
    answer = qa.run(query)
    print(f"Answer: {answer}")