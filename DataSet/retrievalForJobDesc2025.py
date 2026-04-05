import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#Generate your own api key
os.environ['KAGGLE_USERNAME'] = 'username'
os.environ['KAGGLE_KEY'] = 'api-key'

import kaggle
kaggle.api.authenticate()

# Download dataset
kaggle.api.dataset_download_files(
    "adityarajsrv/job-descriptions-2025-tech-and-non-tech-roles",
    path="/tmp/jobs2025",
    unzip=True
)

# See what files downloaded
df = pd.read_csv("/tmp/jobs2025/job_dataset.csv")

# Filter tech jobs by title
tech_keywords = [
    'engineer', 'developer', 'software', 'data', 'analyst',
    'scientist', 'cloud', 'devops', 'machine learning', 'AI',
    'backend', 'frontend', 'fullstack', 'cybersecurity', 'IT',
    'infrastructure', 'python', 'java', 'network', 'database'
]
pattern = '|'.join(tech_keywords)
df_tech = df[df['Title'].str.contains(pattern, case=False, na=False)]

print(f"Total jobs: {len(df)}")
print(f"Tech jobs: {len(df_tech)}")

# Extract skills from both Skills and Keywords columns (semicolon separated)
skills_series = pd.concat([
    df_tech['Skills'].dropna().str.split(';'),
    df_tech['Keywords'].dropna().str.split(';')
]).explode()

skills_series = skills_series.str.strip().str.lower()

# Hard skills whitelist
hard_skills = [
    'python', 'sql', 'java', 'javascript', 'typescript', 'r', 'c++', 'c#',
    'go', 'rust', 'scala', 'kotlin', 'swift', 'php', 'ruby', 'bash', 'perl',
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'spark', 'hadoop',
    'tableau', 'power bi', 'data analysis', 'data visualization', 'nlp',
    'computer vision', 'statistics', 'a/b testing', 'etl',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
    'jenkins', 'ci/cd', 'git', 'github', 'gitlab', 'linux', 'unix',
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
    'snowflake', 'bigquery', 'databricks', 'cassandra',
    'react', 'node.js', 'django', 'flask', 'fastapi', 'rest api', 'graphql',
    'html', 'css', 'vue', 'angular', '.net', 'asp.net', 'entity framework',
    'agile', 'scrum', 'jira', 'microservices', 'api', 'cybersecurity',
    'networking', 'excel', 'microsoft office suite', 'microsoft office'
]

skill_counts = skills_series.value_counts()
hard_skill_counts = skill_counts[skill_counts.index.isin(hard_skills)]

# print("\nTop 30 hard/technical skills:")
# print(hard_skill_counts.head(30))

hard_skill_pct = (hard_skill_counts.head(30) / len(df_tech) * 100).round(1)
hard_skill_pct = hard_skill_pct.astype(str) + '%'

# print("\nTop 30 hard/technical skills (% of tech job postings):")
# print(hard_skill_pct.to_string())


# Split on semicolons and clean each skill — keep as whole phrases
df['Skills_list'] = df['Skills'].str.split(';').apply(
    lambda x: [s.strip().lower() for s in x] if isinstance(x, list) else []
)

# Join skills with a unique separator so TF-IDF treats each skill as one token
# Replace spaces within skills with underscores so "machine learning" → "machine_learning"
df['Skills_clean'] = df['Skills_list'].apply(
    lambda skills: ' '.join([s.replace(' ', '_') for s in skills])
)

# Group by title
title_skills = df.groupby('Title')['Skills_clean'].apply(
    lambda x: ' '.join(x)
).reset_index()

# TF-IDF on whole skill phrases
vectorizer = TfidfVectorizer(
    stop_words=None,        # don't remove stop words — skills like "r" would get removed
    ngram_range=(1, 1),     # single tokens only — each token is already a full skill
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(title_skills['Skills_clean'])
feature_names = vectorizer.get_feature_names_out()

# Convert underscores back to spaces for display
def get_top_terms(title_idx, top_n=5):
    row = tfidf_matrix[title_idx].toarray()[0]
    top_indices = row.argsort()[::-1][:top_n]
    return [(feature_names[i].replace('_', ' '), round(row[i], 4)) 
            for i in top_indices if row[i] > 0]

for idx, row in title_skills.iterrows():
    print(f"\n--- {row['Title']} ---")
    for term, score in get_top_terms(idx):
        print(f"  {term}: {score}")