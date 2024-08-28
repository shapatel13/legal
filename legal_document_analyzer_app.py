import streamlit as st
import google.generativeai as genai
from phi.assistant import Assistant
from phi.llm.google import Gemini
import PyPDF2
import io

# Configure the API with your API key
API_KEY = "AIzaSyB8EQWBGd_L5ISmw01TMDRvZRmTUp5QFJ4"
genai.configure(api_key=API_KEY)

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def analyze_legal_document(pdf_text, query):
    """Analyze a legal document using Gemini model."""
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content using the extracted text
        response = model.generate_content([f"The following is the text extracted from a legal document PDF:\n\n{pdf_text}\n\nBased on this document, {query}"])
        
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create a Legal Document Analyzer Assistant
legal_assistant = Assistant(
    name="Legal Document Analyzer",
    llm=Gemini(model="gemini-1.5-pro", api_key=API_KEY),
    show_tool_calls=True,
    description="You are a legal analyst that specializes in analyzing legal documents and providing insights.",
    instructions=[
        "Analyze legal documents thoroughly and provide detailed insights.",
        "Use legal terminology accurately and explain complex concepts clearly.",
        "Format your response using markdown and use tables to display data where applicable.",
        "Provide citations or references to specific sections of the document when relevant.",
    ],
)

st.title("Legal Document Analyzer")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
query = st.text_input("Enter your legal analysis query:")

if uploaded_file is not None and query:
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if st.button("Analyze Document"):
        with st.spinner("Analyzing the document..."):
            analysis_result = analyze_legal_document(pdf_text, query)
        
        st.subheader("Analysis Result:")
        st.write(analysis_result)
        
        with st.spinner("Generating key legal implications..."):
            assistant_query = f"Based on the following analysis of the legal document, what are the key legal implications? Provide a concise summary:\n\n{analysis_result}"
            legal_implications = legal_assistant.chat(assistant_query)
            
            # Handle the generator object
            implications_text = ""
            for chunk in legal_implications:
                implications_text += chunk
        
        st.subheader("Key Legal Implications:")
        st.markdown(implications_text)

st.markdown("---")
st.markdown("Powered by Gemini and Phi")