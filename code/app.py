#app.py
import os
import time
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Set your Groq API key here
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize your models, databases, and other components here
@st.cache_resource
def init_vectorstore():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_directory = os.path.join(current_dir, "chroma_db")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="my_collection"
    )
    return vectorstore

# Initialize components
client = Groq(api_key=GROQ_API_KEY)
vectorstore = init_vectorstore()

def rag_query(query):
    # Initialize memory buffer and populate it with chat history from session state
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])

    # Retrieve relevant documents using similarity search
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    # Prepare context for LLaMA
    if retrieved_docs:
        context = "\n".join([doc.page_content for doc in retrieved_docs])
    else:
        context = ""

    # Append new interaction to memory
    memory.chat_memory.add_user_message(query)

    # Retrieve past interactions for context
    past_interactions = memory.load_memory_variables({})[memory.memory_key]
    context_with_memory = f"{context}\n\nConversation History:\n{past_interactions}"

    # Debugging: Display context and past interactions
    # st.write("Debugging Info:")
    # st.write("Context Sent to Model:", context_with_memory)
    # st.write("Retrieved Documents:", [doc.page_content for doc in retrieved_docs])
    # st.write("Past Interactions:", past_interactions)

    # Try up to 3 times to handle temporary DNS/connection errors
    for attempt in range(3):
        try:
            # Generate response using llama3-8b-8192
            prompt = f"Context: {context_with_memory}\n\nQuestion: {query}\n\nAnswer:"

            # Get the response from the client
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                max_tokens=500
            )
            response = chat_completion.choices[0].message.content.strip()

            # If the response is empty or very short, or if no relevant documents were found, use the LLM's default knowledge
            if not context or len(response.split()) < 3 or not retrieved_docs:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": query}
                    ],
                    model="llama3-8b-8192",
                    max_tokens=500
                )
                response = chat_completion.choices[0].message.content.strip()
            
            # If successful, break out of the retry loop
            break
        except Exception as e:
            if attempt == 2:
                # On the final attempt, return the error message
                return f"⚠️ Groq API Error: {str(e)}\n\n*(Error details: `{type(e).__name__}`)*"
            # Sleep for 2 seconds before retrying
            time.sleep(2)

    # Append the response to memory
    memory.chat_memory.add_ai_message(response)

    return response

def process_feedback(query, response, feedback):
    if not feedback:
        new_query = f"{query}. Give a better response."
        new_response = rag_query(new_query)
        st.session_state.messages.append(
            {"role": "assistant", "content": new_response}
        )
        st.rerun()

# Streamlit interface
if __name__ == "__main__":
    st.title("Welcome to our RAG-Based Chatbot")
    st.markdown("***")
    st.info('''
            To use Our AI-powered Chatbot, click Chat.
             
            To push data, click on Store Document.
            ''')

    col1, col2 = st.columns(2)

    with col1:
        chat = st.button("Chat")
        if chat:
            st.switch_page("pages/chatbot.py")

    with col2:
        rag = st.button("Store Document")
        if rag:
            st.switch_page("pages/management.py")

    st.markdown("<div style='text-align:center;'></div>", unsafe_allow_html=True)
