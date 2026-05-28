#app.py
import os
import streamlit as st
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
try:
    from langchain.memory import ConversationBufferMemory
except ModuleNotFoundError:
    from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Set your Hugging Face token here
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize your models, databases, and other components here
@st.cache_resource
def init_vectorstore():
    persist_directory = "chroma_db"
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
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)
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

    try:
        # Generate response using LLaMA
        messages = [
            {"role": "user", "content": f"Context: {context_with_memory}\n\nQuestion: {query},it is not mandatory to use the context\n\nAnswer:"}
        ]

        # Get the response from the client
        response_content = client.chat_completion(messages=messages, max_tokens=500, stream=False)

        # Process the response content
        response = response_content.choices[0].message.content.split("Answer:")[-1].strip()

        # If the response is empty or very short, or if no relevant documents were found, use the LLM's default knowledge
        if not context or len(response.split()) < 35 or not retrieved_docs:
            messages = [{"role": "user", "content": query}]
            response_content = client.chat_completion(messages=messages, max_tokens=500, stream=False)
            response = response_content.choices[0].message.content
    except Exception as e:
        if "NameResolutionError" in str(e) or "Failed to resolve" in str(e) or "ConnectionError" in str(e):
            return "⚠️ Connection Error: Failed to connect to the Hugging Face API. Please make sure you are connected to the internet and that 'api-inference.huggingface.co' is not blocked by your network/firewall."
        elif "Authorization" in str(e) or "Unauthorized" in str(e) or "401" in str(e) or "403" in str(e):
            return "🔑 Authentication Error: Your Hugging Face / Mistral token (HF_TOKEN) is invalid or expired. Please check your `.env` file and update your token."
        else:
            return f"⚠️ An error occurred while communicating with the AI model: {e}"

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
            To use Our Mistral supported Chatbot, click Chat.
             
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
