import os
import re
import pandas as pd
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
from docling.document_converter import DocumentConverter
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions

# Fix Hugging Face symlink crash on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
rake = Rake(language='english', sentence_tokenizer=sent_tokenize)

WATSON_API_KEY = "uWh3QMdFnPgNjfpcrBoJM-Sh__2j-WV5zUV772BpbK8n"
WATSON_URL = "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/92560dcd-7ba1-4881-a34e-43b642944bc7"

authenticator = IAMAuthenticator(WATSON_API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator
)
nlu.set_service_url(WATSON_URL)

def mask_pii(text):
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)
    text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{2,5}[-.\s]?\d{3,5}', '[PHONE]', text)
    text = re.sub(r'https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+', '[LINKEDIN]', text, flags=re.I)
    text = re.sub(r'https?://(www\.)?github\.com/[A-Za-z0-9\-_%]+', '[GITHUB]', text, flags=re.I)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = text.replace(ent.text, '[NAME]')
    return text

def extract_top_keywords(text):
    if not isinstance(text, str) or not text.strip():
        return []
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:5]

skill_keywords = [
    # Programming Languages
    'Python', 'Java', 'C', 'C++', 'C#', 'JavaScript', 'TypeScript', 'R', 'Go', 'Kotlin', 'Ruby', 'Swift',
    # Web & App Development
    'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Django', 'Flask', 'Node.js', 'Express', 'Bootstrap',
    # Databases
    'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Oracle', 'Firebase',
    # Cloud Platforms & DevOps
    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'GitHub', 'CI/CD', 'Jenkins',
    # AI / ML / Data Science
    'Machine Learning', 'Deep Learning', 'TensorFlow', 'Keras', 'Scikit-learn', 'PyTorch',
    'Numpy', 'Pandas', 'Matplotlib', 'OpenCV', 'NLTK', 'SpaCy', 'HuggingFace',
    # IoT & Embedded
    'Arduino', 'Raspberry Pi', 'IoT', 'ESP32', 'Arduino IDE', 'MQTT', 'Tinkercad',
    # Networking & Cybersecurity
    'Cisco Packet Tracer', 'Wireshark', 'Nmap', 'Burp Suite', 'Cybersecurity', 'Networking',
    # Tools & Platforms
    'VS Code', 'Jupyter Notebook', 'Git', 'GitHub', 'Figma', 'Blender', 'Unity',
    # Data & Analytics
    'Excel', 'Power BI', 'Tableau', 'SQL', 'Big Data', 'Hadoop', 'Spark',
    # Operating Systems & Environments
    'Linux', 'Ubuntu', 'Windows', 'Shell Scripting', 'Bash'
]  # (keep your full list here)

def extract_resume_info_from_pdf(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    records = []
    for page in doc.texts:
        page_number = page.prov[0].page_no if page.prov else None
        text = page.text
        records.append({"page": page_number, "text": text})

    df = pd.DataFrame(records)
    df['masked_text'] = df['text'].apply(mask_pii)
    resume_text = "\n".join(df['masked_text'].tolist())

    df["keywords"] = df["text"].apply(extract_top_keywords)

    response = nlu.analyze(
        text=resume_text,
        features=Features(
            entities=EntitiesOptions(limit=50),
            keywords=KeywordsOptions(limit=50)
        )
    ).get_result()

    keywords = [k['text'] for k in response.get("keywords", [])]
    entities = [e['text'] for e in response.get("entities", [])]

    skills = list({
        kw for kw in keywords
        if any(skill.lower() in kw.lower() for skill in skill_keywords)
    })

    projects = [line.strip() for line in resume_text.splitlines() if "project" in line.lower()]

    return {
        "skills": skills,
        "entities": list(set(entities)),
        "projects": projects
    }
