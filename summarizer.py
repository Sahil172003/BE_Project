import streamlit as st
import os
import pandas as pd
import json
import sqlite3
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
import tempfile

# Initialize session states
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None

def navigation():
    st.sidebar.title("Navigation")
    pages = {
        "Home": "🏠",
        "CSV/Excel Chat": "📊",
        "JSON Chat": "📝",
        "PDF Chat": "📄",
        "SQL Chat": "💾",
        "Image Generation": "🖼️"
    }
    
    selected_page = st.sidebar.selectbox(
        "Go to",
        list(pages.keys()),
        format_func=lambda x: f"{pages[x]} {x}"
    )
    return selected_page

def home_page():
    st.title("🌟 Welcome to Multi-Document Chat Assistant")
    
    st.markdown("""
    ### 📚 Available Features:
    
    1. **CSV/Excel Chat** 📊
       - Upload and analyze CSV or Excel files
       - Ask questions about your data
    
    2. **JSON Chat** 📝
       - Interact with JSON data
       - Extract insights from JSON structures
    
    3. **PDF Chat** 📄
       - Upload PDF documents
       - Get answers from your PDF content
    
    4. **SQL Database Chat** 💾
       - Connect to SQLite databases
       - Query your data using natural language
       
    5. **Image Generation** 🖼️
       - Generate images from text descriptions
       - Use inpainting to modify existing images
    
    ### 🚀 Getting Started:
    1. Use the sidebar navigation to select your desired feature
    2. Upload your document or database
    3. Start asking questions or creating images!
    """)

def process_file_upload(file, file_type):
    try:
        if file_type == 'csv':
            return pd.read_csv(file)
        elif file_type == 'excel':
            return pd.read_excel(file)
        elif file_type == 'json':
            return json.load(file)
        elif file_type == 'pdf':
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        st.error(f"Error processing {file_type.upper()} file: {str(e)}")
        return None
    
def Imagen():
    st.header("Generate Images with Getimg.ai 🖼️")
    
    import requests  # For API calls
    import io  # For handling image data
    
    # API key for getimg.ai
    API_KEY = "4cUhZ4P3ubyBnKfe82odJ4Pm9jxBqFIlBHy3oimcM5OBPPGDtxxRknXReJRtxhvGOIVXRIzQl18DfDQaDqhVAI2Ex23YzTHo"
    API_URL = "https://api.getimg.ai/v1/flux-schnell/text-to-image"
    
    # Input for text prompt
    prompt = st.text_area("Enter your prompt:", "A beautiful mountain landscape")
    
    # Image size options
    with st.expander("Image Settings"):
        width = st.select_slider("Width", options=[256, 512, 768, 1024], value=512)
        height = st.select_slider("Height", options=[256, 512, 768, 1024], value=512)
    
    # Generate button
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            try:
                # API request payload
                payload = {
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_images": 1  # Number of images to generate
                }
                headers = {
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                }
                
                # Make the API request
                response = requests.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()  # Raise an error for bad responses
                
                # Parse the response
                image_url = response.json()["images"][0]["url"]  # Adjust based on API response structure
                
                # Display the generated image
                st.image(image_url, caption="Generated Image")
                
                # Add download button
                st.download_button(
                    label="Download Image",
                    data=requests.get(image_url).content,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
# def Imagen():
#     st.header("Generate Images with HiDream 🖼️")
    
#     # Import required libraries
#     import torch
#     from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
#     import io
    
#     # Install required packages if needed
#     import subprocess
#     import sys
    
#     @st.cache_resource
#     def install_dependencies():
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes", "accelerate"])
        
#     with st.spinner("Checking dependencies..."):
#         install_dependencies()
    
#     # Input for text prompt
#     prompt = st.text_area("Enter your prompt:", "A beautiful mountain landscape")
    
#     # Quantization options
#     with st.expander("Model Settings"):
#         quantization = st.radio(
#             "Model Quantization Level",
#             ["8-bit (Balanced)", "4-bit (Lowest Memory)"],
#             index=0
#         )
#         model_id = "HiDream-ai/HiDream-I1-Full"
#         image_size = st.select_slider("Image Size", options=[256, 384, 512], value=384)
    
#     # Generate button
#     if st.button("Generate Image"):
#         with st.spinner(f"Loading HiDream model with {quantization}..."):
#             try:
#                 # Set quantization parameters
#                 if "4-bit" in quantization:
#                     load_in_4bit = True
#                     load_in_8bit = False
#                 else:  # 8-bit
#                     load_in_4bit = False
#                     load_in_8bit = True
                
#                 # Load the quantized model - FIX: Remove device_map="auto" or change to "balanced"
#                 pipe = StableDiffusionPipeline.from_pretrained(
#                     model_id,
#                     torch_dtype=torch.float16,
#                     load_in_8bit=load_in_8bit,
#                     load_in_4bit=load_in_4bit,
#                     device_map="balanced",  # Changed from "auto" to "balanced"
#                     safety_checker=None
#                 )
                
#                 # Enable additional memory savings
#                 pipe.enable_attention_slicing()
#                 pipe.enable_vae_slicing()
#                 pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                
#                 st.info("Model loaded successfully. Generating image...")
                
#                 # Generate image
#                 result = pipe(
#                     prompt=prompt,
#                     num_inference_steps=20,
#                     guidance_scale=7.0,
#                     height=image_size,
#                     width=image_size
#                 ).images[0]
                
#                 # Display generated image
#                 st.image(result, caption="Generated Image")
                
#                 # Add download button
#                 buf = io.BytesIO()
#                 result.save(buf, format="PNG")
#                 byte_im = buf.getvalue()
#                 st.download_button(
#                     label="Download Image",
#                     data=byte_im,
#                     file_name="hidream_image.png",
#                     mime="image/png"
#                 )
                
#                 # Free up memory
#                 del pipe
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
            
#             except Exception as e:
#                 st.error(f"Error generating image: {str(e)}")
#                 st.info("Try one of these alternatives: 1) Use smaller image size, 2) Try 4-bit quantization, or 3) Use CPU-only mode")

def csv_excel_chat():
    st.header("Chat with CSV & Excel 📊")
    
    file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])
    
    if file is not None:
        file_extension = file.name.split('.')[-1]
        
        if file_extension == "csv":
            data = process_file_upload(file, 'csv')
        else:
            data = process_file_upload(file, 'excel')
        
        if data is not None:
            st.success("File uploaded successfully!")
            st.dataframe(data.head())
            text = data.to_string(index=False)
            process_text_and_chat(text)

def json_chat():
    st.header("Chat with JSON 📝")
    
    json_file = st.file_uploader("Upload your JSON file", type='json')
    
    if json_file is not None:
        data = process_file_upload(json_file, 'json')
        if data is not None:
            st.success("File uploaded successfully!")
            st.json(data)
            text = json.dumps(data, indent=2)
            process_text_and_chat(text)

def pdf_chat():
    st.header("Chat with PDF 📄")
    
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        text = process_file_upload(pdf, 'pdf')
        if text is not None:
            st.success("File uploaded successfully!")
            with st.expander("View PDF Content"):
                st.text(text[:1000] + "...")
            process_text_and_chat(text)

def sql_chat():
    st.header("Chat with SQLite Database 💾")
    
    db_file = st.file_uploader("Upload your SQLite Database", type='db')
    
    if db_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                tmp_file.write(db_file.getvalue())
                tmp_path = tmp_file.name

            db = SQLDatabase.from_uri(f"sqlite:///{tmp_path}")
            st.success("Database connected successfully!")
            process_sql_chat(db)
            
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing database: {str(e)}")

def process_text_and_chat(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        st.session_state.processed_data = vector_store
        chat_interface()
        
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")

def process_sql_chat(db):
    try:
        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        if not nvidia_api_key:
            st.error("NVIDIA API key not found. Please set the NVIDIA_API_KEY environment variable.")
            return
            
        llm = ChatNVIDIA(model="mixtral_8x7b")
        sql_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
        
        query = st.text_input("Ask questions about your database:", key="sql_query")
        
        if query:
            if st.button("Ask", key="sql_button"):
                with st.spinner("Processing your query..."):
                    response = sql_chain.run(query)
                    st.session_state.chat_history.append({"user": query, "bot": response})
                    
        display_chat_history()
        
    except Exception as e:
        st.error(f"Error in SQL chat: {str(e)}")

def chat_interface():
    query = st.text_input("Ask your question:", key="text_query")
    
    if query:
        if st.button("Ask", key="chat_button"):
            if st.session_state.processed_data is not None:
                try:
                    with st.spinner("Processing your query..."):
                        vector_store = st.session_state.processed_data
                        docs = vector_store.similarity_search(query=query, k=3)
                        
                        nvidia_api_key = os.getenv("NVIDIA_API_KEY")
                        if not nvidia_api_key:
                            st.error("NVIDIA API key not found. Please set the NVIDIA_API_KEY environment variable.")
                            return
                            
                        llm = ChatNVIDIA(model="mixtral_8x7b")
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                        response = chain.run(input_documents=docs, question=query)
                        
                        st.session_state.chat_history.append({"user": query, "bot": response})
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    display_chat_history()

def display_chat_history():
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**👤 You:** {chat['user']}")
                st.markdown(f"**🤖 Assistant:** {chat['bot']}")
                st.markdown("---")

def clear_chat_history():
    st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="Multi-Document Chat Assistant",
        page_icon="🌟",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .css-1d391kg {
            padding: 1rem;
        }
        .stTextInput>div>div>input {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation and page routing
    selected_page = navigation()
    
    # Add clear chat history button in sidebar
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history()
    
    if selected_page == "Home":
        home_page()
    elif selected_page == "CSV/Excel Chat":
        csv_excel_chat()
    elif selected_page == "JSON Chat":
        json_chat()
    elif selected_page == "PDF Chat":
        pdf_chat()
    elif selected_page == "SQL Chat":
        sql_chat()
    elif selected_page == "Image Generation":
        Imagen()

if __name__ == "__main__":
    main()