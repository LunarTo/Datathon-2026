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

# Extract what skills are most prominent in what jobs
df['Skills'] = df['Skills'].str.replace(';', ' ', regex=False)

# Group skills by job title
title_skills = df.groupby('Title')['Skills'].apply(
    lambda x: ' '.join(x.dropna())
).reset_index()

# Apply TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(title_skills['Skills'])
feature_names = vectorizer.get_feature_names_out()

# Top 5 most distinctive skills per title
def get_top_terms(title_idx, top_n=5):
    row = tfidf_matrix[title_idx].toarray()[0]
    top_indices = row.argsort()[::-1][:top_n]
    return [(feature_names[i], round(row[i], 4)) for i in top_indices if row[i] > 0]

for idx, row in title_skills.iterrows():
    print(f"\n--- {row['Title']} ---")
    for term, score in get_top_terms(idx):
        print(f"  {term}: {score}")