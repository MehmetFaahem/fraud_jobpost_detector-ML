import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

class JobFraudDetector:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.stop_words = set(stopwords.words('english'))
        
        self.tfidf_title = TfidfVectorizer(max_features=1000)
        self.tfidf_description = TfidfVectorizer(max_features=1000)
        self.label_encoders = {}
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.fitted_title_vectorizer = None
        self.fitted_desc_vectorizer = None

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def extract_features(self, df, training=False):
        """Extract features from job postings"""
        if training:
            title_features = self.tfidf_title.fit_transform(df['title'].fillna('').apply(self.preprocess_text))
            desc_features = self.tfidf_description.fit_transform(df['description'].fillna('').apply(self.preprocess_text))
            self.fitted_title_vectorizer = self.tfidf_title
            self.fitted_desc_vectorizer = self.tfidf_description
        else:
            title_features = self.fitted_title_vectorizer.transform(df['title'].fillna('').apply(self.preprocess_text))
            desc_features = self.fitted_desc_vectorizer.transform(df['description'].fillna('').apply(self.preprocess_text))

        df['has_email'] = df['description'].fillna('').str.contains(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        df['has_phone'] = df['description'].fillna('').str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        df['has_money_symbol'] = df['description'].fillna('').str.contains(r'\$')
        
        df['company_age'] = 5
        
        categorical_columns = ['employment_type', 'industry', 'location']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].fillna('unknown').unique())
            df[f'{col}_encoded'] = df[col].fillna('unknown').apply(
                lambda x: self.label_encoders[col].transform([x])[0] if x in self.label_encoders[col].classes_ else -1
            )

        feature_matrix = np.hstack([
            title_features.toarray(),
            desc_features.toarray(),
            df[[f'{col}_encoded' for col in categorical_columns]],
            df[['has_email', 'has_phone', 'has_money_symbol', 'company_age']]
        ])
        
        return feature_matrix

    def train(self, df):
        """Train the fraud detection model"""
        X = self.extract_features(df, training=True)
        y = df['fraudulent']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test
    
    def predict(self, job_posting):
        """Predict if a single job posting is fraudulent"""
        df = pd.DataFrame([job_posting])
        X = self.extract_features(df, training=False)
        
        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)
        
        risk_factors = self.get_risk_factors(df)
        
       
        if not risk_factors:
            prediction[0] = 0
        
        return {
            'is_fraudulent': bool(prediction[0]),
            'confidence': float(max(probability[0])),
            'risk_factors': risk_factors
        }
    
    def get_risk_factors(self, df):
        """Identify specific risk factors in the job posting"""
        risk_factors = []
        
        if df['has_email'].iloc[0]:
            risk_factors.append("Contains email address in description")
        if df['has_money_symbol'].iloc[0]:
            risk_factors.append("Contains dollar amounts in description")
        if df['company_age'].iloc[0] < 2:
            risk_factors.append("Company is very new or age unknown")
        if pd.isna(df['requirements'].iloc[0]) or df['requirements'].iloc[0] == '':
            risk_factors.append("Missing job requirements")
            
        return risk_factors