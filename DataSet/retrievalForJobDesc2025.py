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
hard_skills = set([
    # Languages
    'python', 'sql', 'java', 'javascript', 'typescript', 'r', 'c++', 'c#',
    'go', 'rust', 'scala', 'kotlin', 'swift', 'php', 'ruby', 'bash', 'perl',
    'objective-c', 'matlab', 'solidity',

    # Data & ML
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras',
    'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'spark', 'hadoop',
    'tableau', 'power bi', 'powerbi', 'data analysis', 'data visualization', 'nlp',
    'computer vision', 'statistics', 'a/b testing', 'etl',
    'seaborn', 'jupyter', 'logistic regression', 'decision trees',
    'random forest', 'xgboost', 'feature engineering', 'etl automation',
    'etl pipelines', 'reinforcement learning', 'ensemble methods',
    'classification', 'regression', 'time series modeling',
    'predictive modeling', 'data preprocessing', 'k-means',
    'supervised learning', 'unsupervised learning', 'model monitoring',
    'onnx', 'mlflow', 'advanced ml', 'basic ml', 'ml frameworks',
    'ml fundamentals', 'ml libraries', 'basic ml models', 'aws emr',
    'looker studio', 'spss', 'excel', 'google analytics',

    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible',
    'jenkins', 'ci/cd', 'git', 'github', 'gitlab', 'linux', 'unix',
    'cloud networking', 'cloud platforms', 'cloud strategy',
    'cloud cost optimization', 'cloud security', 'serverless architecture',
    'microservices architecture', 'devops automation', 'kubernetes orchestration',
    'aws multi-region', 'azure networking', 'hybrid cloud design',
    'multi-cloud', 'cloudformation', 'cd introduction', 'kubernetes fundamentals',
    'docker basics', 'python scripting', 'nagios', 'grafana', 'prometheus',
    'puppet', 'ansible', 'bash scripting', 'automation scripts',
    'azure fundamentals', 'cloud basics', 'solarwinds',

    # Databases
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
    'snowflake', 'bigquery', 'databricks', 'cassandra', 'sqlite',
    'sqlalchemy', 'room', 'core data', 'nosql', 'sql advanced',

    # Web & APIs
    'react', 'node.js', 'django', 'flask', 'fastapi', 'rest api', 'graphql',
    'html', 'css', 'vue', 'angular', '.net', 'asp.net', 'entity framework',
    'xml', 'json', 'webpack', 'next.js', 'spring boot', 'express.js',
    'tailwind css', 'bootstrap', 'redux', 'restful api design',
    'api integration', 'cypress', 'yarn', 'babel', 'angular basics',
    'react basics', 'js', 'node', 'express', 'restassured', 'postman',
    'soapui', 'selenium', 'selenium webdriver', 'appium', 'testng',
    'jmeter', 'cucumber', 'api testing', 'browser developer tools',

    # Mobile
    'xcode', 'swift', 'kotlin', 'flutter', 'firebase', 'react native',
    'android', 'ios', 'cocoa touch', 'android studio', 'uikit', 'mvvm',
    'mvc', 'interface builder', 'simulator', 'json handling', 'rest apis',
    'jetpack', 'dagger', 'clean architecture', 'room', 'unit testing',
    'performance optimization', 'firebase crashlytics', 'crashlytics',
    'core animation', 'core text', 'coregraphics', 'xcode instruments',
    'performance tuning', 'app store submission', 'ux design', 'ui design',
    'sqlite', 'xml', 'ux basics', 'rest api', 'arkit', 'arcore',

    # UX/UI & Design
    'figma', 'adobe xd', 'sketch', 'invision', 'zeplin', 'wireframing',
    'prototyping', 'user research', 'usability testing', 'ui design',
    'ux design', 'interaction design', 'adobe illustrator', 'adobe photoshop',
    'adobe creative suite', 'canva', 'balsamiq', 'miro',
    'indesign', 'color theory', 'typography', 'motion graphics',
    'information architecture', 'wireframes', 'user flows',
    'layout design', 'basic usability testing', 'user research basics',
    'user research fundamentals', 'prototyping basics', 'axure',
    'journey mapping', 'design thinking', 'accessibility', 'wcag',
    'wcag standards', 'design systems', 'agile ux', 'ux research',
    'after effects', 'principle', 'micro-interactions', 'adobe framemaker',
    'sketch', 'adobe xd', 'b testing', 'accessibility standards',

    # Game Dev
    'unity', 'unreal engine', 'c#', 'opengl', 'vulkan', 'blender',
    'multiplayer networking', 'gazebo', 'solidworks',

    # Cybersecurity
    'cybersecurity', 'penetration testing', 'ethical hacking', 'nessus',
    'metasploit', 'wireshark', 'kali linux', 'siem', 'firewalls', 'vpn',
    'cryptography', 'network security', 'soc', 'burp suite', 'nmap',
    'sqlmap', 'powershell scripting', 'tls', 'snort', 'ids', 'ips',
    'gdpr', 'nist', 'iso 27001', 'qradar', 'splunk', 'logrhythm',
    'threat hunting', 'threat intelligence', 'vulnerability management',
    'incident response', 'malware analysis', 'forensics', 'siem tools',
    'risk management', 'compliance', 'pci dss', 'antivirus',
    'network defense', 'advanced threat detection', 'cloud security',
    'firewall basics', 'gdpr basics', 'nist basics', 'network security basics',
    'linux commands', 'cryptography fundamentals', 'windows security',
    'iso 27001 basics',

    # Networking
    'cisco', 'ospf', 'bgp', 'mpls', 'vpn', 'dns', 'dhcp', 'tcp/ip',
    'lan', 'wan', 'subnetting', 'switching', 'routing', 'cisco ios',
    'cisco routers', 'cisco switches', 'cisco nexus', 'cisco asa',
    'palo alto', 'f5 load balancer', 'network monitoring', 'network troubleshooting',
    'sd-wan', 'zero trust security', 'qos', 'sip', 'voip', 'ip',
    'wan design', 'python automation', 'cisco prime', 'python scripting basics',
    'basic routing', 'network automation', 'vpn basics', 'vpn setup',
    'routing basics', 'bgp advanced', 'ospf advanced', 'architecture design',
    'nagios basics', 'firewalls basics', 'ips intro',

    # Systems & Infrastructure
    'linux', 'windows server', 'active directory', 'vmware', 'hyper-v',
    'virtualbox', 'powershell', 'docker', 'ansible', 'terraform',
    'grafana', 'nagios', 'prometheus', 'aws iam', 'linux security',
    'bash', 'scripting fundamentals', 'monitoring tools', 'networking fundamentals',
    'enterprise architecture', 'virtualization', 'azure', 'aws', 'gcp',
    'system administration', 'cloud basics', 'python basics',

    # Blockchain
    'solidity', 'ethereum', 'web3', 'smart contracts', 'blockchain',
    'blockchain basics',

    # AR/VR & Robotics
    'arkit', 'arcore', 'vuforia', 'openxr', 'webxr',
    'mqtt', 'lorawan', 'iot', 'coap', 'nb-iot',
    'computer vision', 'sensor fusion', 'ros', 'gazebo', 'matlab', 'solidworks',

    # AI
    'prompt engineering', 'langchain', 'openai api', 'hugging face',
    'generative ai', 'llm', 'ai coding assistant', 'debugging ai outputs',

    # Project Management & Collaboration
    'agile', 'scrum', 'jira', 'microservices', 'api', 'networking',
    'microsoft office suite', 'microsoft office', 'confluence',
    'trello', 'notion', 'slack', 'change management', 'conflict resolution',
    'reporting and dashboards', 'kpi monitoring', 'resource allocation',
    'six sigma', 'erp', 'team leadership', 'leadership',

    # Marketing & Sales
    'seo', 'sem', 'email marketing automation', 'brand management',
    'content creation', 'social selling', 'lead generation', 'negotiation',
    'salesforce', 'hubspot', 'market segmentation', 'survey design',
    'presentation skills', 'google ads', 'facebook ads', 'copywriting',
    'content marketing', 'crm', 'marketing analytics',
    'technical seo audits', 'multimodal search optimization',
    'link building', 'privacy', 'google search console', 'ahrefs', 'semrush',

    # Finance & Business
    'financial modeling', 'excel advanced', 'sql advanced',
    'risk analysis', 'data analysis', 'business intelligence',
    'erp systems', 'sap', 'quickbooks', 'accounting software',
    'bloomberg terminal', 'valuation', 'budgeting', 'forecasting',
    'investment analysis',

    # Product Management
    'product roadmap', 'user stories', 'stakeholder management',
    'product strategy', 'competitive analysis', 'go-to-market strategy',
    'okrs', 'kpis', 'product analytics', 'figma', 'jira', 'confluence',
    'tableau', 'ab testing', 'customer discovery', 'market research',

    # Technical Writing & Communication
    'technical writing', 'proofreading', 'madcap flare', 'cms',
    'adobe framemaker', 'css', 'documentation', 'api documentation',
    'markdown', 'confluence', 'notion',

    # Quality Assurance
    'manual testing', 'test documentation', 'functional testing',
    'regression testing', 'postman basics', 'api testing',
    'selenium', 'appium', 'cypress', 'jmeter', 'cucumber', 'testng',
    'restassured', 'soapui', 'performance testing', 'load testing',

    # Operations & HR
    'operations management', 'process improvement', 'supply chain',
    'logistics', 'erp', 'six sigma', 'lean methodology',
    'workforce planning', 'hris', 'applicant tracking systems',
    'performance management', 'employee engagement', 'talent acquisition',
])

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
    stop_words=None,
    ngram_range=(1, 1),
    max_features=5000
)

tfidf_matrix = vectorizer.fit_transform(title_skills['Skills_clean'])
feature_names = vectorizer.get_feature_names_out()

# Convert underscores back to spaces for display, filter to hard skills only
def get_top_terms(title_idx, top_n=5):
    row = tfidf_matrix[title_idx].toarray()[0]
    top_indices = row.argsort()[::-1]
    results = []
    for i in top_indices:
        term = feature_names[i].replace('_', ' ')
        if term in hard_skills and row[i] > 0:
            results.append((term, round(row[i], 4)))
        if len(results) == top_n:
            break
    return results

for idx, row in title_skills.iterrows():
    print(f"\n--- {row['Title']} ---")
    for term, score in get_top_terms(idx):
        print(f"  {term}: {score}")