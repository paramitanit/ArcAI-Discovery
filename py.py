import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


project_requirements_path = 'C:/Users/Paramita.Ghosh/Desktop/Book/project_requirements.csv'
reference_architectures_path = 'C:/Users/Paramita.Ghosh/Desktop/Book/reference_architectures.csv'

project_requirements_df = pd.read_csv(project_requirements_path)
reference_architectures_df = pd.read_csv(reference_architectures_path, encoding='ISO-8859-1')


reference_architectures_df['combined'] = reference_architectures_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


vectorizer = TfidfVectorizer(stop_words='english')
architecture_vectors = vectorizer.fit_transform(reference_architectures_df['combined'])


def find_best_match_for_functionality(project_functionality, reference_architectures_df, architecture_vectors):
    # Vectorize the input functionality
    project_vector = vectorizer.transform([project_functionality])
    
   
    similarity_scores = cosine_similarity(project_vector, architecture_vectors).flatten()
    
  
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


st.title("ArcAI Architecture AI")


project_functionality = st.text_input("Describe your project functionality:")

if project_functionality:
    # Find best matches for the given project functionality
    matches_df = find_best_match_for_functionality(project_functionality, reference_architectures_df, architecture_vectors)
    
  
    pd.set_option('display.max_colwidth', None)
    
  
    matches_df['Reference_Architecture_Description'] = matches_df['Reference_Architecture_Description'].apply(make_urls_clickable)
    

    st.write(matches_df.to_html(escape=False), unsafe_allow_html=True)

