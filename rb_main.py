import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# 1. Setup Page & API Key
st.set_page_config(page_title="Resume Chatbot", page_icon="🤖")
#st.title("Chat with my Resume")
with st.sidebar:
    #st.image("your_profile_picture.jpg", width=150) # If you have one
    st.title("Harsh Resume Bot")
    st.subheader("Bridging the gap between my experience and your questions with AI")
    
    st.markdown("---")
    st.markdown("### 📍 Location")
    st.write("Pune, Maharashtra, India")
    
    st.markdown("### 🔗 Links")
    st.markdown("[LinkedIn] https://www.linkedin.com/in/harsh-dwivedi-14b666204/")
    st.markdown("[GitHub] https://github.com/hdwivedi94")
    
    st.markdown("---")
    st.info("This bot uses Llama 3 and RAG to answer questions based on my official resume.")
    
groq_api_key = st.secrets["GROQ_API_KEY"]

# 2. Initialize RAG Components
@st.cache_resource
def initialize_rag():
    # Load and Split Resume
    loader = PyPDFLoader("Resume.pdf") # Ensure filename matches yours
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = text_splitter.split_documents(data)
    
    # Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

vector_db = initialize_rag()

# 3. Setup LLM & Chat Interface
#llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", google_api_key=gemini_api_key)
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile" # This is a powerful, free model
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_db.as_retriever()
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "clicked_question" not in st.session_state:
    st.session_state.clicked_question = None

# 4. Helper function
def handle_click(question):
    st.session_state.clicked_question = question

# 5. Layout for buttons
st.write("### ⚡ Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("Summarize Profile", on_click=handle_click, args=["Give me a 3-sentence summary of this candidate's career."])
with col2:
    st.button("Top Skills", on_click=handle_click, args=["What are the top 5 technical skills listed in this resume?"])
with col3:
    st.button("Contact Info", on_click=handle_click, args=["How can I contact this person?"])

# 6. Check for input (Either from chat_input OR button click)
user_input = st.chat_input("Ask about my experience...")
prompt = user_input or st.session_state.clicked_question

# 7. Display Chat History (Keeps the UI consistent)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8. Handle the new Prompt
if prompt:
    # Reset the clicked question immediately
    st.session_state.clicked_question = None
    
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): # Added a nice loading spinner
            response = qa_chain.invoke(prompt)
            answer = response["result"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
    # Force a rerun to clear the button state if a button was used
    if not user_input:
        st.rerun()
