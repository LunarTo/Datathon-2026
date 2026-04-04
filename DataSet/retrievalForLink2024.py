#read csv for tech jobs
import os

#Generate your own api key
os.environ['KAGGLE_USERNAME'] = 'username'
os.environ['KAGGLE_KEY'] = 'api-key'

import kaggle
import pandas as pd
kaggle.api.authenticate()

print ("Worked")

#Now do algorithm and extraction

# Download if not already downloaded
kaggle.api.dataset_download_files(
    "asaniczka/1-3m-linkedin-jobs-and-skills-2024",
    path="/tmp/linkedin",
    unzip=True
)

df_jobs = pd.read_csv("/tmp/linkedin/linkedin_job_postings.csv")
df_skills = pd.read_csv("/tmp/linkedin/job_skills.csv")

# Filter tech jobs
tech_keywords = [
    'engineer', 'developer', 'software', 'data', 'analyst',
    'scientist', 'cloud', 'devops', 'machine learning', 'AI',
    'backend', 'frontend', 'fullstack', 'cybersecurity', 'IT',
    'infrastructure', 'python', 'java', 'network', 'database'
]
pattern = '|'.join(tech_keywords)
df_tech = df_jobs[df_jobs['job_title'].str.contains(pattern, case=False, na=False)]

print(f"Total jobs: {len(df_jobs)}")
print(f"Tech jobs: {len(df_tech)}")

# Merge with skills
df_merged = df_tech.merge(df_skills, on="job_link", how="inner")

# Count skills
all_skills = df_merged['job_skills'].dropna().str.split(',').explode()
all_skills = all_skills.str.strip().str.lower()
skill_counts = all_skills.value_counts()

# Hard skills whitelist
hard_skills = [
    # Languages
    'python', 'sql', 'java', 'javascript', 'typescript', 'r', 'c++', 'c#',
    'go', 'rust', 'scala', 'kotlin', 'swift', 'php', 'ruby', 'bash', 'perl',
    # Data & ML
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'spark', 'hadoop',
    'tableau', 'power bi', 'data analysis', 'data visualization', 'nlp',
    'computer vision', 'statistics', 'a/b testing', 'etl',
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
    'jenkins', 'ci/cd', 'git', 'github', 'gitlab', 'linux', 'unix',
    # Databases
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
    'snowflake', 'bigquery', 'databricks', 'cassandra',
    # Web & APIs
    'react', 'node.js', 'django', 'flask', 'fastapi', 'rest api', 'graphql',
    'html', 'css', 'vue', 'angular',
    # Other tech
    'agile', 'scrum', 'jira', 'microservices', 'api', 'cybersecurity',
    'networking', 'excel', 'microsoft office suite', 'microsoft office'
]

# Filter to hard skills only
hard_skill_counts = skill_counts[skill_counts.index.isin(hard_skills)]

print("\nTop 30 hard/technical skills:")
print(hard_skill_counts.head(30))

# Divide by total tech jobs to get percentage
hard_skill_pct = (hard_skill_counts.head(30) / len(df_tech) * 100).round(1)
hard_skill_pct = hard_skill_pct.astype(str) + '%'

print("\nTop 30 hard/technical skills (% of tech job postings):")
print(hard_skill_pct.to_string())