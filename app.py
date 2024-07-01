import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load the project requirements and reference architectures CSV files
project_requirements_path = 'project_requirements.csv'
reference_architectures_path = 'reference_architectures.csv'

project_requirements_df = pd.read_csv(project_requirements_path)
reference_architectures_df = pd.read_csv(reference_architectures_path, encoding='ISO-8859-1')

# Combine all textual fields for vectorization
reference_architectures_df['combined'] = reference_architectures_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Vectorize the textual data
vectorizer = TfidfVectorizer(stop_words='english')
architecture_vectors = vectorizer.fit_transform(reference_architectures_df['combined'])

# Function to find best matching architecture for a given project functionality
def find_best_match_for_functionality(project_functionality, reference_architectures_df, architecture_vectors):
    # Vectorize the input functionality
    project_vector = vectorizer.transform([project_functionality])
    
    # Calculate similarity with reference architectures
    similarity_scores = cosine_similarity(project_vector, architecture_vectors).flatten()
    
    # Find matches with similarity score above 0.1
    matches = []
    for idx, score in enumerate(similarity_scores):
        if score > 0.1:
            best_match_architecture = reference_architectures_df.iloc[idx]
            matches.append({
                'Reference_Architecture_Description': best_match_architecture['Description'],
                'Similarity_Score': score
            })
    
    return pd.DataFrame(matches)

# Function to make URLs clickable in Streamlit
def make_urls_clickable(text):
    url_pattern = re.compile(r'(https?://\S+)')
    return url_pattern.sub(r'<a href="\1" target="_blank">\1</a>', text)

# Streamlit UI
st.title("ArcAI Architecture AI")

# User input
project_functionality = st.text_input("Describe your project functionality:")

if project_functionality:
    # Find best matches for the given project functionality
    matches_df = find_best_match_for_functionality(project_functionality, reference_architectures_df, architecture_vectors)
    
    # Adjust pandas display options to show full text
    pd.set_option('display.max_colwidth', None)
    
    # Convert only URLs in Description to clickable links
    matches_df['Reference_Architecture_Description'] = matches_df['Reference_Architecture_Description'].apply(make_urls_clickable)
    
    # Display results
    st.write(matches_df.to_html(escape=False), unsafe_allow_html=True)

# To run the app, use the command: streamlit run app.py
