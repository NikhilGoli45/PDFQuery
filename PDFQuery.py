from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load the LLM
llm = Ollama(model="llama3.2")

# Load and split document
loader = PyPDFLoader("the_lightning_thief.pdf")
docs = loader.load_and_split()

# Embed and store in vector DB
embedding = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(docs, embedding)

# Create a conversational retrieval chain with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create a retriever and a QA chain
retriever = db.as_retriever(search_kwargs={"k": 20})
qa = qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff",
)

while True:
    # Get user input
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Get the answer from the QA chain
    answer = qa.run(query)
    print(f"Answer: {answer}")