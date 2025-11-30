# Import necessary libraries
import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import email
from email import policy
from langchain.schema import Document
import re
from itertools import islice
import pandas as pd
import plotly.express as px
import json
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
st.set_page_config(page_title="Analytics", page_icon=":bar_chart:", layout="wide")

DATA_PROMPT_TEMPLATE = """
 Extract COMPLETE data from this text. Return EXACTLY:
    {{
        "stocks": [
            {{"name": str, "quantity": float, "percentage": float}},
            ... # ALL entities MUST be included
        ]
    }}

    Rules:
    1. Include EVERY entity mentioned
    2. Put full name
    3. Never return a single object - always use the "stocks" array

    Text: {paragraph}
"""
SUMMARY_PROMPT_TEMPLATE = """
You are an email summarization assistant. 
Provide a clear, concise overall summary of the emails.
Keep the summary to 4-5 sentences. 

Content:
{document_context}

Summary:
"""

ANALYSIS_PROMPT_TEMPLATE = """
You are an expert data analyst. Your task is to strictly analyze the emails below and output a clean list of unique senders, their email counts, and their percentage share.

Instructions:
- Identify unique sender names. Merge duplicates due to casing, spacing, or extra characters.
- Count emails per sender.
- Calculate percentage = (sender_count / total_count) * 100.
- Round percentages to nearest 10% (e.g., ~10%, ~30%) and ensure total is 100%.
- Output only in the format:

Name: <Sender Name>
Quantity: <Count>
Percentage: <~XX%>

Do NOT summarize or explain anything else. No introductions. No bullet points. No reasoning. Just the list.
"""

CLASSIFY_TEMPLATE = """ 

You are an expert in email classification. Your task is to classify the following emails into one of the predefined categories. The categories are:

- **Business**: Emails related to work, professional communication, or business-related topics.
- **Personal**: Emails related to personal communication, friends, family, etc.
- **Spam**: Emails that are promotional, irrelevant, or unsolicited.
- **Social**: Emails related to social media notifications or interactions.
- **Newsletters**: Emails that are newsletters or marketing material.
- **Unknown**: Anything else.

Classify each email below into one of the above categories and output the email’s classification type in a single word. No preamble, postamble, or any explanation or extra word.

Example 1:
Text: "Reminder: Your meeting is scheduled for tomorrow at 9:00 AM."
Output: Business

Example 2:
Text: "Hey! How are you? Let's catch up soon."
Output: Personal

Example 3:
Text: "You’ve won a $500 gift card! Click here to claim it."
Output: Spam

Example 4:
Text: "You have a new message from your friend on Facebook."
Output: Social

Example 5:
Text: "Monthly Newsletter: Top industry news and updates."
Output: Newsletters

Text: {email}

Output:

"""

PRIORITIZE_TEMPLATE = """ 

You are an expert in email prioritization. Your task is to prioritize the following emails based on their urgency. The priority levels are:

- **High Priority**: Critical emails that need immediate attention (e.g., urgent work matters, emergency notifications).
- **Medium Priority**: Important emails that should be reviewed soon (e.g., non-urgent work-related emails, important updates).
- **Low Priority**: Emails that can be addressed later or are not urgent (e.g., newsletters, general information).
- **Unclear**: Emails where the priority cannot be easily determined.

Prioritize the email below based on the descriptions above and output the email's priority level in a single word or two words. No preamble, postamble, or any explanation or extra word.

Example 1:
Text: "Meeting at 9:00 AM tomorrow. Please confirm availability."
Output: High Priority

Example 2:
Text: "Reminder: You have a doctor’s appointment in two days."
Output: Medium Priority

Example 3:
Text: "Reminder: Your subscription to XYZ Magazine is expiring."
Output: Low Priority

Example 4:
Text: "Emergency: Your bank account has been compromised. Please act immediately."
Output: High Priority

Example 5:
Text: "Are you free for lunch tomorrow?"
Output: Unclear

Text: {email}

Output:
"""

SAVED_PATH = 'document_store/mails/'
PDF_STORAGE_PATH = 'document_store/mails/'
EMBEDDING_MODEL = OllamaEmbeddings(model="llama3.2")  #deepseek-r1:1.5b / 
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = Ollama(model="llama3.2")

#Deep seek model
EMBEDDING_MODEL_D = OllamaEmbeddings(model="mistral:7b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL_D)
LANGUAGE_MODEL_D = Ollama(model="mistral:7b") # phi3:14b
# LANGUAGE_MODEL_D = Ollama(model="deepseek-r1:1.5b", temperature=0.1)

def parse_email_file(file_input):
    try:
        if hasattr(file_input, "read"):  # Streamlit uploaded file
            raw_data = file_input.read()
            name = file_input.name
        else:  # Local file path
            with open(file_input, "rb") as f:
                raw_data = f.read()
            name = file_input

        msg = email.message_from_bytes(raw_data, policy=policy.default)
        body = ""

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            body = msg.get_content()

        metadata = {
            "from": msg.get("From"),
            "to": msg.get("To"),
            "subject": msg.get("Subject"),
            "date": msg.get("Date"),
            "filename": name
        }

        return Document(page_content=body.strip(), metadata=metadata)

    except Exception as e:
        st.warning(f"Could not parse file: {file_input} – {e}")
        return None

def save_uploaded_files(uploaded_files):
    saved_paths = []

    # Ensure it's a list (in case user sends a single file)
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for uploaded_file in uploaded_files:
        if hasattr(uploaded_file, 'name'):
            file_path = PDF_STORAGE_PATH + uploaded_file.name
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
        else:
            print(f"Skipping invalid file object: {uploaded_file}")

    return saved_paths

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

@st.cache_resource
def generate_analysis(_document_chunks):
    context_text = "\n\n".join([doc.page_content for doc in _document_chunks])  # use all chunks
    analysis_prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT_TEMPLATE)
    response_chain = analysis_prompt | LANGUAGE_MODEL_D
    return response_chain.invoke({"document_context": context_text})

@st.cache_resource
def generate_summary(_document_chunks):
    context_text = "\n\n".join([doc.page_content for doc in _document_chunks])  # use all chunks
    summary_prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
    response_chain = summary_prompt | LANGUAGE_MODEL_D
    return response_chain.invoke({"document_context": context_text})

@st.cache_resource
def generate_data(document_chunks):
    data_prompt = ChatPromptTemplate.from_template(DATA_PROMPT_TEMPLATE)
    response_chain = data_prompt | LANGUAGE_MODEL | JsonOutputParser()
    response = response_chain.invoke({"paragraph": document_chunks})
    return response

@st.cache_resource
def visualize_stocks(data):
    """Generate interactive visualizations for stock data"""
    if isinstance(data, dict) and 'stocks' in data:
        df = pd.json_normalize(data['stocks'])
    else:
        df = pd.DataFrame(data)
    figures = []
    print("Available columns:", df.columns.tolist())
    # 1. Allocation Pie Chart
    if 'percentage' in df.columns:
        fig1 = px.pie(
            df,
            names='name',
            values='percentage',
            title='Analysis (%)',
            hover_data=['quantity'],
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_traces(
            textinfo='percent+label',  # CHANGED: 'name' → 'label'
            insidetextorientation='radial',
            textposition='inside'
        )
        figures.append(fig1)
    
    if 'quantity' in df.columns:
        fig2 = px.bar(
        df.sort_values('quantity', ascending=False),
        x='name',
        y='quantity',
        color='name',
        text='quantity',
        title='Share Quantities',
        labels={'quantity': 'Shares/Units', 'name': 'Person'},
        color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig2.update_traces(
            textposition='outside',
            texttemplate='%{text:,}'
        )
        figures.append(fig2)
        
    # 3. Combined Treemap
    if len(df) > 2:
        fig3 = px.treemap(
            df,
            path=['name'],
            values='percentage' if 'percentage' in df.columns else 'quantity',
            title='Distribution',
            color='quantity',
            color_continuous_scale='Blues'
        )
        fig3.update_traces(
            textinfo='label+value',
            texttemplate='%{label}<br>%{value}%' if 'percentage' in df.columns else '%{label}<br>%{value}'
        )
        figures.append(fig3)
    
    return figures

@st.cache_resource
def split_emails(text):
    # Pattern to match email headers (From, Subject, Date, etc.)
    email_pattern = r"From:.*?Subject:.*?(?=From:|\Z)"  
    # Find all matching email blocks
    emails = re.findall(email_pattern, text, flags=re.DOTALL)
    return emails

@st.cache_resource
def classify(email):
    # context_text = "\n\n".join([doc.page_content for doc in _document_chunks])  # use all chunks
    classify_prompt = ChatPromptTemplate.from_template(CLASSIFY_TEMPLATE)
    response_chain = classify_prompt | LANGUAGE_MODEL_D
    raw_eclass = response_chain.invoke({"email": email})
    eclass = re.sub(r'<think>.*?</think>', '', raw_eclass, flags=re.DOTALL).strip()
    return eclass

@st.cache_resource
def prioritize(email):
    prioritize_prompt = ChatPromptTemplate.from_template(PRIORITIZE_TEMPLATE)
    response_chain = prioritize_prompt | LANGUAGE_MODEL_D
    raw_priority = response_chain.invoke({"email": email})
    priority = re.sub(r'<think>.*?</think>', '', raw_priority, flags=re.DOTALL).strip()
    return priority

@st.cache_resource
def load_docs():
    existing_docs = []
    max_files_to_process = 1
    with st.spinner("Loading existing emails..."):
        for filename in os.listdir(SAVED_PATH):
            if len(existing_docs) >= max_files_to_process:
                break
            file_path = os.path.join(SAVED_PATH, filename)
            if os.path.isfile(file_path):
                doc = parse_email_file(file_path)
                if doc:
                    existing_docs.append(doc)

    raw_docs = existing_docs
    processed_chunks = chunk_documents(raw_docs)
    st.session_state.processed_chunks = processed_chunks  # ✅ store it
    index_documents(processed_chunks)
    return processed_chunks

with st.spinner("calling function..."):
     processed_chunks = load_docs()
if "passchunks" not in st.session_state:
    st.session_state.passchunks = processed_chunks
#MUTED
raw_analysis = generate_analysis(processed_chunks)
auto_analysis = re.sub(r'<think>.*?</think>', '', raw_analysis, flags=re.DOTALL).strip()
st.session_state.auto_analysis = auto_analysis
input_text = st.session_state.auto_analysis

with st.spinner("Analyzing Emails..."):
    stocks = generate_data(input_text)
    figures = visualize_stocks(stocks)

raw_summary = generate_summary(processed_chunks)
auto_summary = re.sub(r'<think>.*?</think>', '', raw_summary, flags=re.DOTALL).strip()
context_text = "\n\n".join([doc.page_content for doc in processed_chunks])
emails = split_emails(context_text)

if 'categories' not in st.session_state:
    categories = {}
    for email in emails:
        category = classify(email)  
        if category not in categories:
            categories[category] = []  
        categories[category].append(email)

    st.session_state.categories = categories

if 'priorities' not in st.session_state:
    priorities = {}
    for email in emails:
        priority = prioritize(email)  
        if priority not in priorities:
            priorities[priority] = []  
        priorities[priority].append(email) 
    
    # Store the prioritized emails in session state
    st.session_state.priorities = priorities

if "chart" not in st.session_state:
    st.session_state.chart = figures
    st.session_state.summary = auto_summary

home_page = st.Page(
    "views/home.py",
    title="Home",
    icon=":material/home:",
    default=True,
)
project_1_page = st.Page(
    "views/classify.py",
    title="Classification",
    icon=":material/density_medium:",
)
project_2_page = st.Page(
    "views/prioritize.py",
    title="Priority",
    icon=":material/label_important:",
)
project_3_page = st.Page(
    "views/chatbot.py",
    title="Chatbot",
    icon=":material/smart_toy:",
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Overview": [home_page],
        "Menu": [project_1_page, project_2_page, project_3_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assets/email.png")
st.sidebar.markdown("Deal with your emails like a pro! 📧🤖")


# --- RUN NAVIGATION ---
pg.run()
