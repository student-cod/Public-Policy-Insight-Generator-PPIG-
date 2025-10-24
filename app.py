# app.py: Public Policy Insight Generator (PPIG) - FINAL VERSION

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import nltk
import io
from pypdf import PdfReader # New import for PDF handling

# --- CRITICAL IMPORTS ---
# Ensure 'config.py' exists with GEMINI_API_KEY = "YOUR_KEY"
from config import GEMINI_API_KEY 
from google import genai
from google.genai.errors import APIError

# --- VADER MODEL SETUP ---
# Fixes the 'DownloadError' by checking and downloading VADER lexicon data
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception: 
    nltk.download('vader_lexicon', quiet=True)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- DATA EXTRACTION HELPERS ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file uploaded via Streamlit."""
    try:
        # Read file into a bytes stream
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        raw_text = ""
        
        # Concatenate text from all pages
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
        
        # Return as a DataFrame row for consistent processing
        return pd.DataFrame({'feedback': [raw_text]})

    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
        return None

def load_data(uploaded_file):
    """Loads and preprocesses the uploaded file (CSV or PDF)."""
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    df = None
    
    try:
        if file_extension == 'csv':
            # --- CSV LOGIC (Includes robust 'latin-1' encoding fix) ---
            df = pd.read_csv(uploaded_file, encoding='latin-1') 
            
        elif file_extension == 'pdf':
            # --- PDF LOGIC ---
            df = extract_text_from_pdf(uploaded_file)
            
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or PDF.")
            return None
            
        if df is None: return None
        
        # Clean column names and check for required 'feedback' column
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        if 'feedback' not in df.columns:
            st.error("❌ Error: File must result in a column named 'feedback'.")
            return None
        return df

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# --- 1. MODEL LOADING (VADER) ---
@st.cache_resource
def load_vader_model():
    """Loads the VADER Sentiment Analyzer instantly."""
    st.info("⏳ Initializing fast VADER Sentiment Model...")
    analyzer = SentimentIntensityAnalyzer()
    st.success("✅ VADER Sentiment Model Loaded Instantly!")
    return analyzer

# --- 2. VADER ANALYSIS & AGGREGATION LOGIC ---
def run_full_analysis(df, analyzer):
    """Performs General Sentiment Analysis (VADER) and aggregates results."""
    
    st.subheader("2. Performing General Sentiment Analysis (Fast VADER)")
    
    if not analyzer or df.empty:
        return pd.DataFrame(), 0

    start_time = time.time()
    
    df['vader_scores'] = df['feedback'].apply(lambda x: analyzer.polarity_scores(x))
    
    def get_vader_sentiment(scores):
        if scores['compound'] >= 0.05:
            return 'POSITIVE'
        elif scores['compound'] <= -0.05:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
            
    df['General_Sentiment'] = df['vader_scores'].apply(get_vader_sentiment)
    
    st.success(f"✅ VADER Analysis Complete! Time: {time.time() - start_time:.2f}s")
    
    # --- AGGREGATION ---
    sentiment_counts = df['General_Sentiment'].value_counts().to_dict()
    total = len(df)
    
    positive = sentiment_counts.get('POSITIVE', 0)
    negative = sentiment_counts.get('NEGATIVE', 0)
    
    summary_df = pd.DataFrame({
        'Sentiment Type': ['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
        'Count': [positive, negative, sentiment_counts.get('NEUTRAL', 0)]
    })
    
    overall_net_sentiment = (positive - negative) / total if total > 0 else 0
    
    return summary_df, overall_net_sentiment

# --- 3. GENERATIVE AI CHATBOT LOGIC ---
def generate_policy_report(summary_df, overall_net_sentiment, aspect_labels):
    """Connects to Gemini to generate the actionable advice."""
    
    st.subheader("4. Policy Advisor Chatbot (Action Plan)")
    
    report_data_string = summary_df.to_markdown(index=False)
    
    prompt = f"""
    You are a Senior Policy Advisor. A public feedback analysis showed the following:
    
    1. Overall Net Sentiment Score (VADER): {overall_net_sentiment:.2f} (1.0 is max positive).
    2. Sentiment Counts:\n{report_data_string}
    3. The policy aspects the public is concerned with are: {aspect_labels}

    Your task is to provide an actionable report. 
    
    RULES:
    1. Describe the overall public mood (Positive, Negative, or Mixed) based on the Net Score.
    2. Suggest three highly probable areas of public concern based on the *Aspect Labels* provided 
       (Cost, Safety, Traffic, etc.) and generate a concise 3-point plan to address the mood.
    
    Generate the response using professional markdown format with clear headings.
    """

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        with st.spinner("🧠 Gemini is synthesizing policy advice..."):
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            st.markdown(response.text)
            
    except APIError:
        st.error("❌ Gemini API Error: The policy advice could not be generated. Please check your API key or usage limits.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")

# --- 4. MAIN STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Policy Insight Generator (PPIG)", layout="wide")
    st.title("🏛️ Public Policy Insight Generator (PPIG)")
    st.markdown("Analyze public feedback using AI to identify policy strengths and weaknesses.")

    # --- Sidebar Input ---
    with st.sidebar:
        st.header("1. Data & Controls")
        # Updated uploader to accept both CSV and PDF
        uploaded_file = st.file_uploader(
            "Upload Policy Feedback File (CSV or PDF)",
            type=["csv", "pdf"],
            help="For PDF, all text will be treated as one single document."
        )
        
        analyzer = load_vader_model() 
        
        aspect_input = st.text_area("Define Policy Aspects (comma-separated):", "Cost, Safety, Traffic, Timeline")
        run_analysis = st.button("Run Aspect Analysis")
        
        aspect_labels = [label.strip() for label in aspect_input.split(',')]

    if uploaded_file is None:
        st.info("☝️ Please upload a file to begin.")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    st.header("1. Raw Data Preview")
    st.write(f"Loaded {len(df)} rows.")
    st.dataframe(df.head())

    # --- EXECUTE ANALYSIS ---
    if run_analysis and analyzer:
        
        summary_df, overall_net_sentiment = run_full_analysis(df.copy(), analyzer)
        
        if summary_df.empty:
            return

        # --- Visualization ---
        st.subheader("3. Policy Insight Dashboard")
        
        st.markdown(f"##### Overall Net Sentiment Score: **{overall_net_sentiment:.2f}**")
        st.dataframe(summary_df.set_index('Sentiment Type'), use_container_width=True)
        
        # Plotly Chart
        st.markdown("##### Sentiment Distribution Plot")
        fig = px.bar(
            summary_df, x='Sentiment Type', y='Count', color='Sentiment Type', 
            color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Generative AI Call ---
        generate_policy_report(summary_df, overall_net_sentiment, aspect_labels)

if __name__ == "__main__":
    main()