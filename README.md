# 📧 AI Email Assistant (Streamlit)

A multi-page Streamlit web app for intelligent email analysis, classification, prioritization, and chat-based Q&A. Built with Python and modern AI libraries.

## Features
Dashboard: Visualize email categories and priorities with interactive charts.
Classification: Automatically sort emails into Business, Personal, Spam, Social, Newsletters, or Unknown.
Prioritization: Assign urgency levels to emails (High, Medium, Low, Unclear).
Chatbot: Ask questions about your emails and get AI-powered answers.
Modern UI: Clean, responsive design with custom styling.

## Technologies Used
Python 3.11+
Streamlit
Pandas
Plotly
LangChain
Ollama (LLM/Embeddings)
Getting Started
Clone the repository:

## Install dependencies:
pip install -r requirements.txt

## Run the app
```Powershell
# vanilla terminal
streamlit run main.py

# quit
ctrl-c
```

## Project Structure
main.py
requirements.txt
assets/
    email.png
document_store/
    mails/
        Dataset.txt
views/
    chatbot.py
    classify.py
    home.py
    prioritize.py

## Usage
Upload your email files or use the provided dataset.
Navigate between pages for analytics, classification, prioritization, and chat.
Get actionable insights and summaries from your emails.




