import os
import tempfile
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, List
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import streamlit as st

# --- Configuration and Initialization ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide", # Changed layout for better spacing
    initial_sidebar_state="expanded" # Keep sidebar open for file upload
)
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("🚨 Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], lambda x, y: x + y] # Correct way to append
    context: str
    input: str


@st.cache_resource(show_spinner="Initializing AI components...") # Cache resources
def init_components():
    """Initializes LLM and Embeddings models."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Using flash as it's often sufficient and faster/cheaper
            google_api_key=os.environ["GOOGLE_API_KEY"],
            temperature=0.7, # Add some temperature for variability
            convert_system_message_to_human=True # Good practice for some models
        )
        return embeddings, llm
    except Exception as e:
        st.error(f"🚨 Error initializing AI components: {e}")
        st.stop() # Stop execution if core components fail

embeddings, llm = init_components()

def process_uploaded_files(uploaded_files):
    """Loads, splits, and prepares documents from uploaded files."""
    all_docs = []
    if not uploaded_files:
        return []

    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            file_ext = os.path.splitext(uploaded_file.name)[1].lower()

            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_ext == ".docx":
                loader = Docx2txtLoader(tmp_file_path)
            elif file_ext == ".txt":
                loader = TextLoader(tmp_file_path, encoding='utf-8') # Specify encoding
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                os.remove(tmp_file_path) # Clean up temp file
                continue

            docs = loader.load()
            all_docs.extend(docs)
            os.remove(tmp_file_path) # Clean up temp file after loading

        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

    if not all_docs:
        return []

    # 2. Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(all_docs)

@st.cache_resource(show_spinner="Embedding documents...") # Cache vector store creation
def create_vector_store(_docs: List[Document], _embeddings):    # Use leading _ to indicate unused args for caching
    """Creates a Chroma vector store from documents."""
    if not _docs:
        st.warning("No documents processed to create vector store.")
        return None
    try:
        return Chroma.from_documents(documents=_docs, embedding=_embeddings)
    except Exception as e:
        st.error(f"🚨 Error creating vector store: {e}")
        return None

# --- LangGraph Nodes ---

def retrieve(state: AgentState):
    """Retrieves relevant documents from the vector store."""
    print("--- Retrieving ---")
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.error("Vector store not initialized. Please upload documents.")
        return {"context": "", "messages": state["messages"], "input": state["input"]}

    vector_store = st.session_state.vector_store
    query = state["input"]
    print(f"Retrieval query: {query}")

    try:
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        print(f"Retrieved context (first 100 chars): {context[:100]}...")
        return {"context": context, "messages": state["messages"], "input": state["input"]}
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return {"context": "Error during retrieval.", "messages": state["messages"], "input": state["input"]}

def generate(state: AgentState):
    """Generates an answer using the LLM based on context and history."""
    print("--- Generating ---")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based *only* on the provided context. If the context doesn't contain the answer, state that clearly. Be concise and informative. Do not make up information. Context:\n{context}"),
        ("human", "{question}")
    ])

    question = state["input"]
    context = state["context"]

    messages = prompt_template.format_messages(context=context, question=question)
    print(f"LLM input messages: {messages}")

    try:
        response = llm.invoke(messages)
        print(f"LLM response: {response.content}")
        # IMPORTANT: Only return fields to *update* the state.
        # LangGraph handles appending AIMessage based on the state definition.
        return {"messages": [AIMessage(content=response.content)]}
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        # Consider logging the full error `traceback.format_exc()`
        # Return a helpful error message to the user via the state
        error_message = "Sorry, I encountered an error while generating the response."
        return {"messages": [AIMessage(content=error_message)]}


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

try:
    app = workflow.compile()
except Exception as e:
    st.error(f"🚨 Error compiling LangGraph workflow: {e}")
    st.stop()

# --- Streamlit UI ---

st.title("📚 RAG Chatbot with LangGraph")
st.markdown("Upload PDF, DOCX, or TXT documents, then ask questions about their content.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload documents using the sidebar to get started."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None # Flag to check if docs are processed

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("📁 Document Upload")
    if uploaded_files := st.file_uploader(
        "Upload your documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader",  
    ):
        if st.button("Process Documents", key="process_button"):
            with st.spinner("Processing documents... This may take a moment."):
                if split_docs := process_uploaded_files(uploaded_files):
                    st.session_state.vector_store = create_vector_store(split_docs, embeddings)
                    if st.session_state.vector_store:
                        st.success(f"✅ Successfully processed {len(uploaded_files)} document(s). Ready to chat!")
                        st.session_state.messages = [{"role": "assistant", "content": "Documents processed! Ask me anything about them."}]
                    else:
                        st.error("🚨 Failed to create vector store after processing documents.")
                else:
                    st.warning("⚠️ No text could be extracted from the uploaded documents.")
                    st.session_state.vector_store = None # Ensure it's None if processing failed


# --- Chat Interface ---

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "Ask a question about the document(s)...",
    # Disable input if documents haven't been processed successfully
    disabled=not st.session_state.vector_store,
):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    current_state: AgentState = {
        "messages": [HumanMessage(content=prompt)], # Start with the current user prompt
        "input": prompt,
        "context": "",
    }

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                final_state = app.invoke(current_state)
                response_message = final_state['messages'][-1]
                if isinstance(response_message, AIMessage):
                    answer = response_message.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Received an unexpected message type from the AI.")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, an internal error occurred."})

            except Exception as e:
                st.error(f"An error occurred during chat processing: {e}")
                error_msg = "Sorry, I ran into a problem. Please try again."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
