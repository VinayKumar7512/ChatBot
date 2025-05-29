from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import re
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

class MedicalMLChatbot:
    def __init__(self):
        self.disclaimer = "⚠️ This is for informational purposes only. Always consult a healthcare professional."
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
        
        self.symptom_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.disease_classifier = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        self.medicine_classifier = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
        
        self.label_encoders = {}
        self.is_trained = False
        
        # Medical knowledge base
        self.medical_data = self.create_training_data()
        
        # Symptom keywords for multi-symptom detection
        self.symptom_keywords = {
            'headache': ['headache', 'head pain', 'migraine', 'head ache', 'cranial pain'],
            'fever': ['fever', 'high temperature', 'temperature', 'hot', 'feverish', 'pyrexia'],
            'cold': ['cold', 'runny nose', 'stuffy nose', 'congestion', 'sniffles', 'rhinitis'],
            'cough': ['cough', 'coughing', 'hack', 'throat clearing'],
            'sore_throat': ['sore throat', 'throat pain', 'scratchy throat', 'throat ache'],
            'nausea': ['nausea', 'sick', 'queasy', 'stomach upset', 'feel sick'],
            'vomiting': ['vomit', 'vomiting', 'throw up', 'throwing up'],
            'diarrhea': ['diarrhea', 'loose stools', 'watery stool', 'runs'],
            'fatigue': ['tired', 'fatigue', 'exhausted', 'weakness', 'weak'],
            'body_aches': ['body aches', 'muscle pain', 'aching', 'sore muscles'],
            'chills': ['chills', 'shivering', 'cold sweats', 'shaking'],
            'loss_of_appetite': ['no appetite', 'loss of appetite', 'not hungry'],
            'dizziness': ['dizzy', 'dizziness', 'lightheaded', 'vertigo'],
            'chest_pain': ['chest pain', 'chest discomfort', 'heart pain'],
            'abdominal_pain': ['stomach pain', 'belly pain', 'abdominal pain', 'tummy ache']
        }
        
        # Enhanced greeting and interaction patterns
        self.greeting_patterns = {
            'hello': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
            'how_are_you': ['how are you', 'how do you do', 'how you doing', 'whats up', "what's up"],
            'thanks': ['thank you', 'thanks', 'thank u', 'ty', 'appreciate', 'grateful'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'later'],
            'help': ['help', 'assist', 'support', 'guide', 'what can you do'],
            'about': ['what are you', 'who are you', 'tell me about yourself', 'about you']
        }
        
        self.greeting_responses = {
            'hello': [
                "👋 Hello! I'm your AI Medical Assistant. How can I help you with your health concerns today?",
                "🏥 Hi there! I'm here to help analyze your symptoms and provide medical guidance. What's troubling you?",
                "✨ Greetings! I'm your medical AI companion. Please tell me about any symptoms you're experiencing.",
                "🌟 Hello! Welcome to your personal health assistant. How are you feeling today?",
                "💊 Hi! I'm ready to help you understand your symptoms better. What brings you here today?"
            ],
            'how_are_you': [
                "🤖 I'm functioning well and ready to help! More importantly, how are YOU feeling? Any health concerns?",
                "💪 I'm operating at full capacity! But let's focus on you - how is your health today?",
                "🎯 I'm doing great, thank you for asking! Now, tell me about your well-being. Any symptoms bothering you?",
                "🌈 I'm excellent and eager to assist! What about you? Are you experiencing any health issues?",
                "⚡ I'm running smoothly! The real question is - how are you feeling physically? Any concerns?"
            ],
            'thanks': [
                "🙏 You're very welcome! I'm glad I could help. Feel free to ask if you have more health questions!",
                "😊 Happy to help! Remember, take care of your health and don't hesitate to consult a doctor when needed.",
                "💝 My pleasure! Stay healthy and remember that professional medical care is always important.",
                "🌟 Anytime! Your health is important. Please seek professional help for serious concerns.",
                "❤️ Glad to assist! Remember to follow up with healthcare professionals for proper treatment."
            ],
            'goodbye': [
                "👋 Take care and stay healthy! Remember to consult healthcare professionals for serious concerns. Goodbye!",
                "🌟 Farewell! Wishing you good health. Don't forget - always see a doctor for proper medical care!",
                "💚 Goodbye! Take care of yourself and remember that professional medical advice is irreplaceable!",
                "🏥 See you later! Stay safe and healthy. Professional medical consultation is always recommended!",
                "✨ Bye! Remember to prioritize your health and seek medical help when needed. Take care!"
            ],
            'help': [
                "🆘 I can help you with:\n• Analyzing your symptoms\n• Suggesting possible conditions\n• Recommending treatments\n• Providing health information\n\nJust describe your symptoms!",
                "📋 Here's what I can assist you with:\n• Multi-symptom analysis\n• Disease identification\n• Treatment suggestions\n• Health guidance\n\nTell me what's bothering you!",
                "🔍 I'm designed to:\n• Understand complex symptom combinations\n• Provide medical insights\n• Suggest appropriate treatments\n• Offer health advice\n\nWhat symptoms are you experiencing?",
                "💡 I can help by:\n• Analyzing your health concerns\n• Identifying potential conditions\n• Recommending medicines\n• Providing care instructions\n\nDescribe your symptoms to get started!"
            ],
            'about': [
                "🤖 I'm an AI Medical Assistant powered by machine learning! I can analyze symptoms, suggest conditions, and recommend treatments. I use advanced ML models to understand complex symptom patterns. How can I help you today?",
                "🏥 I'm your intelligent health companion! I'm trained on medical data to help identify symptoms, diseases, and treatments. I specialize in multi-symptom analysis. What health concerns do you have?",
                "💊 I'm an advanced medical AI that combines multiple machine learning models to provide health insights. I can handle complex symptom combinations and provide comprehensive medical guidance. Tell me your symptoms!",
                "🔬 I'm a sophisticated medical chatbot using Random Forest and Logistic Regression models. I'm designed to understand your health concerns and provide informed guidance. What's troubling you today?"
            ]
        }
        
        self.encouragement_phrases = [
            "Don't worry, I'm here to help! 💪",
            "Let's figure this out together! 🤝",
            "I'll do my best to assist you! ⭐",
            "You came to the right place for help! 🎯",
            "I'm here to guide you through this! 🧭"
        ]
        
        self.casual_responses = {
            'feeling_bad': [
                "I'm sorry to hear you're not feeling well. 😔 Let me help you understand what might be going on.",
                "That doesn't sound fun! 💙 Let's work together to figure out what's happening.",
                "I understand you're going through a tough time. 🤗 I'm here to help you feel better."
            ],
            'multiple_symptoms': [
                "It sounds like you're dealing with several symptoms. 🔍 That's exactly what I'm designed to help with!",
                "Multiple symptoms can be concerning, but you're in good hands! 👨‍⚕️ Let me analyze everything together.",
                "I see you have various symptoms - perfect! 🎯 I specialize in understanding complex symptom patterns."
            ]
        }
        
        # Train the models
        self.train_models()

    def is_greeting_only(self, text):
        text_lower = text.lower().strip()
        # Remove punctuation for better matching
        text_clean = re.sub(r'[^\w\s]', '', text_lower)
        # Pure greeting patterns (no medical context)
        pure_greetings = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'good night', 'how are you', 'how do you do', 
            'how you doing', 'whats up', "what's up", 'thank you', 'thanks', 
            'thank u', 'ty', 'thanks a lot', 'thank you so much', 'appreciate it',
            'bye', 'goodbye', 'see you', 'farewell', 'take care', 'later', 
            'help', 'what can you do', 'who are you', 'what are you', 
            'tell me about yourself', 'about you'
        ]
    
        # Check if the entire message is just a greeting (with or without punctuation)
        for greeting in pure_greetings:
            if text_clean == greeting or text_lower == greeting:
                return True
        # Also check with common punctuation
            if text_lower in [greeting + '!', greeting + '.', greeting + '?']:
                return True
            simple_greetings = ['hi there', 'hello there', 'hey there', 'good day']
            if text_clean in simple_greetings or text_lower in simple_greetings:
                return True
            return False
        # Check for simple greeting combinations
        simple_greetings = ['hi there', 'hello there', 'hey there', 'good day', 'thanks so much']
        for greeting in simple_greetings:
            if text_clean == greeting or text_lower == greeting:
                return True
            
        # Check if it's just "thank you" with variations
        thank_variations = ['thank you', 'thanks', 'thank u', 'ty', 'thx', 'tysm', 'thank you so much', 'thanks a lot']
        if text_clean in thank_variations:
            return True
        
        return False

    def has_medical_context(self, text):
        """Check if the text contains medical symptoms or health-related content"""
        text_lower = text.lower()
        
        # Check for symptom keywords
        for symptom_list in self.symptom_keywords.values():
            for keyword in symptom_list:
                if keyword in text_lower:
                    return True
        
        # Check for medical context words
        medical_words = [
            'pain', 'hurt', 'ache', 'sick', 'ill', 'unwell', 'symptom', 'symptoms',
            'disease', 'condition', 'treatment', 'medicine', 'medication', 'drug',
            'doctor', 'hospital', 'clinic', 'health', 'medical', 'diagnose',
            'infection', 'virus', 'bacteria', 'allergy', 'allergic', 'prescription',
            'feeling bad', 'not feeling well', 'feel terrible', 'feel awful'
        ]
        
        for word in medical_words:
            if word in text_lower:
                return True
                
        return False

    def detect_interaction_type(self, text):
        """Detect the type of user interaction (greeting, question, etc.)"""
        text_lower = text.lower().strip()
        
        for interaction_type, patterns in self.greeting_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return interaction_type
        return None

    def get_random_response(self, response_type):
        """Get a random response from the specified type"""
        if response_type in self.greeting_responses:
            return random.choice(self.greeting_responses[response_type])
        return None

    def add_personality_touch(self, response, symptom_count=0):
        """Add personality and encouragement to responses"""
        if symptom_count > 2:
            encouragement = random.choice(self.casual_responses['multiple_symptoms'])
            response = f"{encouragement}\n\n{response}"
        elif any(word in response.lower() for word in ['sorry', 'concern', 'worry']):
            encouragement = random.choice(self.casual_responses['feeling_bad'])
            response = f"{encouragement}\n\n{response}"
        
        return response

    def get_time_based_greeting(self):
        """Generate time-appropriate greeting"""
        current_hour = datetime.now().hour
        
        if 5 <= current_hour < 12:
            time_greeting = "Good morning! 🌅"
        elif 12 <= current_hour < 17:
            time_greeting = "Good afternoon! ☀️"
        elif 17 <= current_hour < 22:
            time_greeting = "Good evening! 🌆"
        else:
            time_greeting = "Good night! 🌙"
        
        return f"{time_greeting} I'm your AI Medical Assistant. How can I help you today?"

    def extract_symptoms_from_text(self, text):
        """Extract multiple symptoms from user input"""
        text_lower = text.lower()
        detected_symptoms = []
        
        for symptom, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if symptom not in detected_symptoms:
                        detected_symptoms.append(symptom)
                    break
        
        return detected_symptoms

    def create_training_data(self):
        """Create comprehensive training dataset with multi-symptom combinations"""
        
        # Enhanced symptoms data with multi-symptom scenarios
        symptoms_data = [
            # Single symptoms
            ("I have a severe headache", "headache", "migraine", "sumatriptan"),
            ("My head is pounding", "headache", "tension_headache", "ibuprofen"),
            ("I have high temperature", "fever", "viral_infection", "acetaminophen"),
            ("Running a fever", "fever", "bacterial_infection", "antibiotics"),
            ("I have a cold", "cold", "common_cold", "decongestants"),
            ("Runny nose and congestion", "cold", "rhinitis", "antihistamines"),
            ("Persistent cough", "cough", "bronchitis", "cough_suppressants"),
            ("Dry cough", "cough", "allergies", "antihistamines"),
            
            # Multi-symptom combinations - Cold & Flu
            ("I have cold, fever and headache", "cold+fever+headache", "flu", "flu_medication"),
            ("Cold symptoms with fever", "cold+fever", "viral_infection", "symptomatic_treatment"),
            ("Fever, headache and body aches", "fever+headache+body_aches", "flu", "analgesics"),
            ("Runny nose, cough and fever", "cold+cough+fever", "upper_respiratory_infection", "combination_cold_medicine"),
            ("Headache, fever and chills", "headache+fever+chills", "flu", "flu_treatment"),
            ("Cold, cough and sore throat", "cold+cough+sore_throat", "viral_upper_respiratory", "throat_lozenges"),
            ("Fever, cough and fatigue", "fever+cough+fatigue", "respiratory_infection", "rest_and_fluids"),
            ("Headache, nausea and fever", "headache+nausea+fever", "viral_illness", "antiemetics"),
            
            # Gastrointestinal combinations
            ("Nausea, vomiting and fever", "nausea+vomiting+fever", "gastroenteritis", "oral_rehydration"),
            ("Stomach pain and diarrhea", "abdominal_pain+diarrhea", "gastroenteritis", "probiotics"),
            ("Fever, vomiting and headache", "fever+vomiting+headache", "viral_gastroenteritis", "antiemetics"),
            ("Abdominal pain and nausea", "abdominal_pain+nausea", "gastritis", "antacids"),
            
            # Respiratory combinations
            ("Cough, fever and chest pain", "cough+fever+chest_pain", "pneumonia", "antibiotics"),
            ("Shortness of breath and cough", "dyspnea+cough", "respiratory_infection", "bronchodilators"),
            ("Chest pain and difficulty breathing", "chest_pain+dyspnea", "respiratory_distress", "emergency_care"),
            
            # Comprehensive flu-like symptoms
            ("Fever, headache, body aches and fatigue", "fever+headache+body_aches+fatigue", "influenza", "oseltamivir"),
            ("Cold, fever, cough and headache", "cold+fever+cough+headache", "flu", "combination_flu_medicine"),
            ("Runny nose, fever, headache and tiredness", "cold+fever+headache+fatigue", "viral_syndrome", "symptomatic_care"),
            
            # Other combinations
            ("Headache and dizziness", "headache+dizziness", "tension_headache", "pain_relievers"),
            ("Fever and chills", "fever+chills", "infection", "fever_reducers"),
            ("Fatigue and body aches", "fatigue+body_aches", "viral_infection", "rest_and_pain_relief"),
        ]
        
        return symptoms_data

    def prepare_training_data(self):
        """Prepare data for ML training with enhanced multi-symptom support"""
        texts = []
        symptoms = []
        diseases = []
        medicines = []
        
        for text, symptom, disease, medicine in self.medical_data:
            # Create more variations of the text
            variations = [
                text,
                text.replace("I have", "I'm experiencing"),
                text.replace("I have", "I feel"),
                text.replace("I have", "I've got"),
                text.replace("pain", "discomfort"),
                text.replace("severe", "intense"),
                text.replace("and", ","),
                f"What causes {symptom.replace('+', ' and ')}?",
                f"How to treat {symptom.replace('+', ' and ')}?",
                f"Medicine for {symptom.replace('+', ' and ')}",
                f"Symptoms of {disease}",
                f"Treatment for {disease}",
                f"I'm suffering from {symptom.replace('+', ' and ')}",
                f"Help me with {symptom.replace('+', ' and ')}"
            ]
            
            for variation in variations:
                texts.append(variation)
                symptoms.append(symptom)
                diseases.append(disease)
                medicines.append(medicine)
        
        return texts, symptoms, diseases, medicines

    def train_models(self):
        """Train all ML models"""
        print("Training ML models...")
        
        # Prepare training data
        texts, symptoms, diseases, medicines = self.prepare_training_data()
        
        # Encode labels
        self.label_encoders['symptoms'] = LabelEncoder()
        self.label_encoders['diseases'] = LabelEncoder()
        self.label_encoders['medicines'] = LabelEncoder()
        
        symptom_labels = self.label_encoders['symptoms'].fit_transform(symptoms)
        disease_labels = self.label_encoders['diseases'].fit_transform(diseases)
        medicine_labels = self.label_encoders['medicines'].fit_transform(medicines)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train symptom classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X, symptom_labels, test_size=0.2, random_state=42
        )
        self.symptom_classifier.fit(X_train, y_train)
        symptom_accuracy = accuracy_score(y_test, self.symptom_classifier.predict(X_test))
        
        # Train disease classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X, disease_labels, test_size=0.2, random_state=42
        )
        self.disease_classifier.fit(X_train, y_train)
        disease_accuracy = accuracy_score(y_test, self.disease_classifier.predict(X_test))
        
        # Train medicine classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X, medicine_labels, test_size=0.2, random_state=42
        )
        self.medicine_classifier.fit(X_train, y_train)
        medicine_accuracy = accuracy_score(y_test, self.medicine_classifier.predict(X_test))
        
        self.is_trained = True
        
        print(f"Model Training Complete!")
        print(f"Symptom Classifier Accuracy: {symptom_accuracy:.3f}")
        print(f"Disease Classifier Accuracy: {disease_accuracy:.3f}")
        print(f"Medicine Classifier Accuracy: {medicine_accuracy:.3f}")
        
        # Save models
        self.save_models()

    def save_models(self):
        """Save trained models to disk"""
        try:
            models = {
                'vectorizer': self.vectorizer,
                'symptom_classifier': self.symptom_classifier,
                'disease_classifier': self.disease_classifier,
                'medicine_classifier': self.medicine_classifier,
                'label_encoders': self.label_encoders
            }
            
            with open('medical_models.pkl', 'wb') as f:
                pickle.dump(models, f)
            print("Models saved successfully!")
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            with open('medical_models.pkl', 'rb') as f:
                models = pickle.load(f)
            
            self.vectorizer = models['vectorizer']
            self.symptom_classifier = models['symptom_classifier']
            self.disease_classifier = models['disease_classifier']
            self.medicine_classifier = models['medicine_classifier']
            self.label_encoders = models['label_encoders']
            self.is_trained = True
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict(self, user_input):
        """Make predictions using trained models with multi-symptom analysis"""
        if not self.is_trained:
            return None
        
        try:
            # Extract individual symptoms from input
            detected_symptoms = self.extract_symptoms_from_text(user_input)
            
            # Vectorize input
            input_vector = self.vectorizer.transform([user_input])
            
            # Get predictions with probabilities
            symptom_pred = self.symptom_classifier.predict(input_vector)[0]
            symptom_proba = self.symptom_classifier.predict_proba(input_vector)[0].max()
            
            disease_pred = self.disease_classifier.predict(input_vector)[0]
            disease_proba = self.disease_classifier.predict_proba(input_vector)[0].max()
            
            medicine_pred = self.medicine_classifier.predict(input_vector)[0]
            medicine_proba = self.medicine_classifier.predict_proba(input_vector)[0].max()
            
            # Decode predictions
            predicted_symptom = self.label_encoders['symptoms'].inverse_transform([symptom_pred])[0]
            predicted_disease = self.label_encoders['diseases'].inverse_transform([disease_pred])[0]
            predicted_medicine = self.label_encoders['medicines'].inverse_transform([medicine_pred])[0]
            
            return {
                'symptom': {
                    'name': predicted_symptom,
                    'confidence': float(symptom_proba)
                },
                'disease': {
                    'name': predicted_disease,
                    'confidence': float(disease_proba)
                },
                'medicine': {
                    'name': predicted_medicine,
                    'confidence': float(medicine_proba)
                },
                'detected_symptoms': detected_symptoms  # Individual symptoms found
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_comprehensive_medical_info(self, symptom_combination, disease, medicine, individual_symptoms):
        """Get detailed medical information for multiple symptoms"""
        
        # Enhanced medical information database
        medical_info = {
            'symptom_combinations': {
                'cold+fever+headache': {
                    'description': 'Common flu-like symptoms indicating viral infection',
                    'all_symptoms': ['runny/stuffy nose', 'elevated body temperature', 'head pain'],
                    'common_causes': ['influenza virus', 'common cold virus', 'respiratory infection'],
                    'when_to_see_doctor': 'High fever (>103°F), severe headache, difficulty breathing, symptoms lasting >10 days'
                },
                'fever+headache+body_aches': {
                    'description': 'Classic flu symptoms with systemic involvement',
                    'all_symptoms': ['elevated temperature', 'head pain', 'muscle aches'],
                    'common_causes': ['influenza', 'viral infection', 'bacterial infection'],
                    'when_to_see_doctor': 'Severe symptoms, high fever, chest pain, difficulty breathing'
                },
                'cold+cough+fever': {
                    'description': 'Upper respiratory infection symptoms',
                    'all_symptoms': ['nasal congestion', 'persistent cough', 'fever'],
                    'common_causes': ['viral upper respiratory infection', 'bronchitis', 'sinusitis'],
                    'when_to_see_doctor': 'Persistent high fever, blood in cough, severe chest pain'
                }
            },
            'diseases': {
                'flu': {
                    'description': 'Viral infection affecting respiratory system with systemic symptoms',
                    'symptoms': ['fever', 'headache', 'body aches', 'fatigue', 'cough', 'sore throat'],
                    'treatment': 'Rest, fluids, antiviral medications, symptom management',
                    'duration': '7-10 days',
                    'complications': 'Pneumonia, bronchitis, sinus infections'
                },
                'common_cold': {
                    'description': 'Mild viral infection primarily affecting nose and throat',
                    'symptoms': ['runny nose', 'sneezing', 'mild cough', 'low-grade fever'],
                    'treatment': 'Rest, fluids, decongestants, pain relievers',
                    'duration': '7-10 days',
                    'complications': 'Usually none in healthy individuals'
                },
                'viral_infection': {
                    'description': 'General viral illness with variable symptoms',
                    'symptoms': ['fever', 'fatigue', 'body aches', 'respiratory symptoms'],
                    'treatment': 'Supportive care, rest, fluids, symptom management',
                    'duration': '5-14 days',
                    'complications': 'Secondary bacterial infections'
                }
            },
            'medicines': {
                'flu_medication': {
                    'components': ['Acetaminophen/Ibuprofen for fever and pain', 'Decongestants for congestion', 'Cough suppressants'],
                    'type': 'Combination therapy',
                    'uses': 'Multi-symptom relief for flu-like illness',
                    'dosage': 'Follow package instructions for each component',
                    'warnings': 'Do not exceed recommended doses, avoid alcohol, consult doctor if symptoms worsen'
                },
                'combination_cold_medicine': {
                    'components': ['Pain reliever', 'Decongestant', 'Antihistamine', 'Cough suppressant'],
                    'type': 'Multi-symptom cold relief',
                    'uses': 'Treats multiple cold symptoms simultaneously',
                    'dosage': 'As directed on package, typically every 4-6 hours',
                    'warnings': 'May cause drowsiness, avoid driving, do not combine with other medications containing same ingredients'
                },
                'symptomatic_treatment': {
                    'components': ['Fever reducers', 'Pain relievers', 'Decongestants', 'Rest and fluids'],
                    'type': 'Supportive care approach',
                    'uses': 'Manages symptoms while body fights infection',
                    'dosage': 'Individual medications as needed',
                    'warnings': 'Monitor symptoms, seek medical care if worsening'
                }
            }
        }
        
        result = {
            'individual_symptoms': individual_symptoms,
            'symptom_count': len(individual_symptoms)
        }
        
        # Add symptom combination info
        if symptom_combination in medical_info['symptom_combinations']:
            result['symptom_combination_info'] = medical_info['symptom_combinations'][symptom_combination]
        
        # Add disease info
        if disease in medical_info['diseases']:
            result['disease_info'] = medical_info['diseases'][disease]
        
        # Add medicine info
        if medicine in medical_info['medicines']:
            result['medicine_info'] = medical_info['medicines'][medicine]
        
        return result

    def get_response(self, user_input):
        """Generate chatbot response with proper greeting handling"""
        user_input = user_input.strip()
        
        if not user_input:
            return self.get_time_based_greeting()
        
        # CRITICAL FIX: Check if it's a pure greeting first, before any medical analysis
        if self.is_greeting_only(user_input):
            interaction_type = self.detect_interaction_type(user_input)
            if interaction_type:
                greeting_response = self.get_random_response(interaction_type)
                if greeting_response:
                    return greeting_response
            # Fallback for greetings not in patterns
            return self.get_time_based_greeting()
        if not self.has_medical_context(user_input):
            return ("🤔 I didn't detect any health-related symptoms in your message. \n\n" +
                    "I'm here to help with medical concerns! Try describing symptoms like:\n" +
                    "• 'I have a headache and fever'\n" +
                    "• 'I'm feeling nauseous'\n" +
                    "• 'I have a cough and sore throat'\n\n" +
                    "What health symptoms are you experiencing? 💙")
        # Check for emergency keywords
        emergency_keywords = ['emergency', 'urgent', 'chest pain', 'difficulty breathing', 'severe pain', 'can\'t breathe']
        if any(keyword in user_input.lower() for keyword in emergency_keywords):
            return "🚨 **MEDICAL EMERGENCY DETECTED!** 🚨\n\nIf this is a medical emergency, call emergency services (911) immediately! Don't wait - get professional help right now!\n\nI'm just an AI assistant and cannot replace emergency medical care."
        
        # Extract individual symptoms
        detected_symptoms = self.extract_symptoms_from_text(user_input)
        
        # Check if user is expressing feeling unwell without specific symptoms
        feeling_bad_keywords = ['not feeling well', 'feeling sick', 'unwell', 'feeling bad', 'not good']
        if any(keyword in user_input.lower() for keyword in feeling_bad_keywords) and not detected_symptoms:
            return ("😔 I'm sorry to hear you're not feeling well! \n\n" +
                   "To help you better, could you please describe your specific symptoms? For example:\n" +
                   "• Do you have a fever, headache, or body aches?\n" +
                   "• Are you experiencing nausea, cough, or fatigue?\n" +
                   "• Any other discomfort or pain?\n\n" +
                   "The more details you provide, the better I can assist you! 💙")
        
        # Make ML predictions
        predictions = self.predict(user_input)
        
        if not predictions:
            return ("🤔 I'm having trouble understanding your symptoms. Let me help you describe them better!\n\n" +
                   "Try mentioning specific symptoms like:\n" +
                   "• 'I have a headache and fever'\n" +
                   "• 'I'm experiencing nausea and stomach pain'\n" +
                   "• 'I have a cough, runny nose, and feel tired'\n\n" +
                   "Feel free to describe multiple symptoms - I'm designed to understand complex combinations! 🎯")
        
        # Check confidence thresholds
        min_confidence = 0.2
        if (predictions['symptom']['confidence'] < min_confidence and 
            predictions['disease']['confidence'] < min_confidence):
            return ("🤷‍♀️ I'm not very confident about my analysis of your symptoms. This could mean:\n\n" +
                   "• Your symptoms might be uncommon or complex\n" +
                   "• You might need to describe them more specifically\n" +
                   "• Professional medical evaluation would be most helpful\n\n" +
                   "💡 Try rephrasing your symptoms or consider consulting a healthcare provider for the best guidance!\n\n" + 
                   self.disclaimer)
        
        # Get comprehensive medical information
        medical_info = self.get_comprehensive_medical_info(
            predictions['symptom']['name'],
            predictions['disease']['name'],
            predictions['medicine']['name'],
            detected_symptoms
        )
        
        # Format response for multiple symptoms
        response = "**🔍 Multi-Symptom Analysis:**\n\n"
        
        # Show individual detected symptoms
        if detected_symptoms:
            response += f"**Detected Symptoms ({len(detected_symptoms)}):**\n"
            for i, symptom in enumerate(detected_symptoms, 1):
                response += f"{i}. {symptom.replace('_', ' ').title()}\n"
            response += "\n"
        
        # Add combined symptom analysis
        response += f"**Symptom Pattern:** {predictions['symptom']['name'].replace('_', ' ').replace('+', ' + ').title()}\n\n"
        
        # Add disease info
        response += f"**Likely Condition:** {predictions['disease']['name'].replace('_', ' ').title()}\n\n"
        
        # Add treatment recommendation
        response += f"**Recommended Treatment:** {predictions['medicine']['name'].replace('_', ' ').title()}\n\n"
        
        # Add detailed medical information
        if 'symptom_combination_info' in medical_info:
            info = medical_info['symptom_combination_info']
            response += f"**About Your Symptoms:**\n"
            response += f"• {info['description']}\n"
            response += f"• Common causes: {', '.join(info['common_causes'])}\n"
            response += f"• See a doctor if: {info['when_to_see_doctor']}\n\n"
        
        if 'disease_info' in medical_info:
            info = medical_info['disease_info']
            response += f"**Condition Details:**\n"
            response += f"• {info['description']}\n"
            response += f"• Typical duration: {info.get('duration', 'Variable')}\n"
            response += f"• Treatment approach: {info['treatment']}\n\n"
        
        if 'medicine_info' in medical_info:
            info = medical_info['medicine_info']
            response += f"**Treatment Information:**\n"
            if 'components' in info:
                response += f"• Contains: {', '.join(info['components'])}\n"
            response += f"• Purpose: {info['uses']}\n"
            response += f"• Important: {info['warnings']}\n\n"
        
        response += "**General Care Recommendations:**\n"
        response += "• Get plenty of rest and sleep\n"
        response += "• Stay well hydrated with water and clear fluids\n"
        response += "• Eat light, nutritious foods as tolerated\n"
        response += "• Monitor your temperature regularly\n"
        response += "• Avoid close contact with others to prevent spread\n\n"
        
        response += "**Important Reminders:**\n"
        response += "• This AI analysis considers multiple symptoms together\n"
        response += "• Always consult healthcare professionals for proper diagnosis\n"
        response += "• Seek immediate care if symptoms worsen or new concerning symptoms develop\n\n"
        response += self.disclaimer
        
        return response

# Initialize chatbot
print("Initializing Enhanced Medical ML Chatbot...")
chatbot = MedicalMLChatbot()

# Try to load existing models, if not available, train new ones
if not chatbot.load_models():
    print("No existing models found. Training new models...")
    chatbot.train_models()

@app.route('/')
def serve_home():
    """Serve the HTML interface"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>File Not Found</title></head>
        <body>
            <h1>Error: index.html file not found</h1>
            <p>Please make sure the index.html file is in the same directory as chatbot.py</p>
        </body>
        </html>
        ''', 404

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handle chat messages with enhanced multi-symptom ML predictions"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        # Get ML-based response with multi-symptom analysis
        bot_response = chatbot.get_response(user_message)
        
        # Get raw predictions for frontend
        predictions = chatbot.predict(user_message) if user_message.strip() else None
        
        return jsonify({
            'response': bot_response,
            'predictions': predictions,
            'model_status': 'trained' if chatbot.is_trained else 'not_trained'
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'response': 'Sorry, I encountered an error. Please try again.',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the trained models"""
    try:
        if not chatbot.is_trained:
            return jsonify({'status': 'not_trained'})
        
        info = {
            'status': 'trained',
            'total_symptoms': len(chatbot.label_encoders['symptoms'].classes_),
            'total_diseases': len(chatbot.label_encoders['diseases'].classes_),
            'total_medicines': len(chatbot.label_encoders['medicines'].classes_),
            'training_samples': len(chatbot.medical_data) * 14,  # More variations now
            'model_types': {
                'symptom_classifier': 'Random Forest',
                'disease_classifier': 'Logistic Regression', 
                'medicine_classifier': 'Random Forest'
            },
            'features': [
                'Multi-symptom detection',
                'Symptom combination analysis', 
                'Enhanced training data',
                'Keyword-based symptom extraction'
            ]
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🏥 Enhanced Medical ML Chatbot Backend Started!")
    print("📊 Multi-symptom analysis models trained and ready")
    print("🔍 Now supports complex symptom combinations")
    print("🌐 Access the chatbot at: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)