import pandas as pd
from job_fraud_detector import JobFraudDetector

def analyze_job_description(text):
    """Analyze a job description text and determine if it's potentially fraudulent"""
    job_posting = {
        'title': extract_title(text),
        'description': text,
        'requirements': extract_requirements(text),
        'company_profile': extract_company_info(text),
        'location': extract_location(text),
        'salary_range': extract_salary(text),
        'employment_type': 'Full-time',
        'industry': 'Technology',
        'benefits': '',
        'fraudulent': None
    }
    
    detector = JobFraudDetector()
    df = pd.read_csv('job_postings.csv')
    detector.train(df)
    
    result = detector.predict(job_posting)
    return result

def extract_title(text):
    """Extract job title from the text"""
    first_lines = text.split('\n')
    for line in first_lines:
        if line.strip() and not line.lower().startswith('about'):
            return line.strip()
    return "Position Not Specified"

def extract_requirements(text):
    """Extract requirements section from the text"""
    requirement_keywords = [
        "requirements", "qualifications", "skills", "what we're looking for", "what we are looking for"
    ]
    for keyword in requirement_keywords:
        if keyword.lower() in text.lower():
            sections = text.lower().split(keyword.lower())
            if len(sections) > 1:
                next_section = sections[1].split('\n\n')[0]
                return next_section.strip()
    return ""

def extract_company_info(text):
    """Extract company information from the text"""
    company_keywords = ["about", "company", "who we are", "our story"]
    for keyword in company_keywords:
        if keyword.lower() in text.lower():
            sections = text.lower().split(keyword.lower())
            if len(sections) > 1:
                company_info = sections[1].split('\n\n')[0]
                return company_info.strip()
    return ""

def extract_location(text):
    """Extract location from the text"""
    location_keywords = ["location", "based in", "located in"]
    for keyword in location_keywords:
        if keyword.lower() in text.lower():
            sections = text.lower().split(keyword.lower())
            if len(sections) > 1:
                location_info = sections[1].split('\n')[0]
                return location_info.strip()
    return "Unknown"

def extract_salary(text):
    """Extract salary from the text"""
    salary_keywords = ["salary", "compensation", "pay"]
    for keyword in salary_keywords:
        if keyword.lower() in text.lower():
            sections = text.lower().split(keyword.lower())
            if len(sections) > 1:
                salary_info = sections[1].split('\n')[0]
                return salary_info.strip()
    return "Unknown"

if __name__ == "__main__":
    job_description = '''SA-based startup revolutionizing safety-critical systems development for industries like automotive, aerospace, and healthcare. By leveraging AI and advanced formal methods, we automate complex processes to reduce costs, improve safety, and enable faster innovation.
Requirements: Relevant experience in AI and formal methods
This is not your typical AI job. At RoboFication, you’ll work on research-driven projects that tackle real-world challenges in critical industries, offering an unparalleled opportunity to learn and grow.
'''
    result = analyze_job_description(job_description)

    print("\nPrediction Results:")
    print(f"Is Fraudulent: {result['is_fraudulent']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nRisk Factors:")
    for factor in result['risk_factors']:
        print(f"- {factor}")