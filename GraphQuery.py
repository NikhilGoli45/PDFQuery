from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from typing import TypedDict, List, Tuple, Optional

def setup_vectorstore(pdf_path="the_lightning_thief.pdf"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(split_docs, embedding)
    db.save_local("faiss_index")  # optional: save for reuse
    return db

def retrieve_documents(state):
    query = state["question"]
    retriever = state["retriever"]
    chat_history = state.get("chat_history", [])
    full_query = " ".join([q for q, _ in chat_history]) + " " + query

    docs = retriever.get_relevant_documents(full_query.strip())
    return {**state, "documents": docs}

def generate_answer(state):
    docs = state["documents"]
    llm = state["llm"]
    query = state["question"]

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Answer the question using ONLY the context below.

Context:
{context}

Question: {query}"""

    answer = llm.invoke(prompt)
    return {**state, "answer": answer}

def update_history(state):
    question = state["question"]
    answer = state["answer"]
    chat_history = state.get("chat_history", [])
    chat_history.append((question, answer))
    return {**state, "chat_history": chat_history}

class ChatState(TypedDict):
    question: str
    chat_history: List[Tuple[str, str]]
    retriever: object
    llm: object
    documents: Optional[list]
    answer: Optional[str]

def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("retrieve", RunnableLambda(retrieve_documents))
    builder.add_node("generate", RunnableLambda(generate_answer))
    builder.add_node("update_history", RunnableLambda(update_history))

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "update_history")
    builder.set_finish_point("update_history")

    return builder.compile()


def main():
    # Load or build vectorstore
    try:
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        db = FAISS.load_local("faiss_index", embedding)
    except:
        db = setup_vectorstore()

    retriever = db.as_retriever(search_kwargs={"k": 20})
    llm = Ollama(model="llama3.2")
    graph = build_graph()

    chat_history = []

    print("Ask me anything about the book. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("exit", "quit"):
            break

        state = {
            "question": question,
            "chat_history": chat_history,
            "retriever": retriever,
            "llm": llm,
        }

        result = graph.invoke(state)
        answer = result["answer"]
        chat_history = result["chat_history"]

        print("AI:", answer)

if __name__ == "__main__":
    main()
