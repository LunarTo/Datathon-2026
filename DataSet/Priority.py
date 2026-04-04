#doing weighted priority formula

#independent frequencies, taken from scraping with the other files
jobs_2025 = {
    'python': 0.523, 'sql': 0.314, 'aws': 0.280, 'java': 0.244,
    'docker': 0.235, 'ci/cd': 0.235, 'git': 0.222, 'kubernetes': 0.175,
    'azure': 0.159, 'tableau': 0.153, 'node.js': 0.152, 'r': 0.143,
    'power bi': 0.140, 'agile': 0.137, 'javascript': 0.133,
    'machine learning': 0.130, 'c#': 0.113, 'react': 0.105,
    'terraform': 0.095, 'gcp': 0.094, 'mongodb': 0.093, 'c++': 0.087,
    'django': 0.085, 'tensorflow': 0.083, 'hadoop': 0.082,
    'angular': 0.081, 'html': 0.080, 'postgresql': 0.076,
    'microservices': 0.075, 'mysql': 0.075
}

linkedin_2024 = {
    'data analysis': 0.082, 'microsoft office suite': 0.053,
    'python': 0.052, 'microsoft office': 0.044, 'sql': 0.042,
    'excel': 0.035, 'java': 0.029, 'aws': 0.027, 'agile': 0.021,
    'javascript': 0.019, 'networking': 0.018, 'machine learning': 0.017,
    'linux': 0.016, 'kubernetes': 0.015, 'azure': 0.015, 'docker': 0.014,
    'data visualization': 0.014, 'c++': 0.014, 'git': 0.012, 'jira': 0.012,
    'tableau': 0.012, 'c#': 0.010, 'html': 0.009, 'power bi': 0.009,
    'statistics': 0.009, 'scrum': 0.008, 'css': 0.008, 'r': 0.008,
    'go': 0.008, 'react': 0.008
}

# Weights
weightA = 0.5   
weightB = 0.5   

# Make 2025 stronger than 2024
recency_2024 = 0.8
recency_2025 = 1.0

# Building scores
all_skills = set(linkedin_2024.keys()) | set(jobs_2025.keys())

results = {}
for skill in all_skills:
    score_2024 = linkedin_2024.get(skill, 0)
    score_2025 = jobs_2025.get(skill, 0)

    # Weighted average of frequency scores
    freq_score = (score_2024 + score_2025) / 2

    # Recency weighted score (2025 counts more)
    time_score = (score_2024 * recency_2024 + score_2025 * recency_2025) / 2

    priority = (freq_score * weightA) + (time_score * weightB)
    results[skill] = round(priority, 4)

# Sort and display
#Again independent numbers
import pandas as pd
df_priority = pd.DataFrame(results.items(), columns=['skill', 'priority'])
df_priority = df_priority.sort_values('priority', ascending=False).reset_index(drop=True)

print(df_priority.to_string(index=False))