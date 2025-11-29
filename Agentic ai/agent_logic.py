import os
import json
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables. Please set it.")

genai.configure(api_key=GOOGLE_API_KEY)

# Constants
MODEL_NAME = "gemini-1.5-pro" # Using 1.5 Pro as the robust reasoning model
MAX_TURNS = 6
CONFIDENCE_THRESHOLD = 0.75
SCORE_DIFF_THRESHOLD = 0.12

class AgentState(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    history: Dict[str, Any] = Field(default_factory=dict) # lifestyle, family history, etc.
    risk_scores: Dict[str, float] = Field(default_factory=dict)
    turn_count: int = 0
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    is_finished: bool = False
    final_report: Optional[Dict[str, Any]] = None

class FollowUpQuestion(BaseModel):
    type: str = "followup"
    question: str
    feature_target: str
    options: List[str]

class FinalReport(BaseModel):
    type: str = "final_report"
    detected_condition: str
    probability: float
    symptom_trigger_report: str
    exercise_meditation_yoga_plan: List[Dict[str, Any]]
    safe_medicine_info: List[Dict[str, Any]]
    motivation: str
    disclaimer: str

DISEASES = [
    "Diabetes",
    "Hypertension",
    "Heart Disease",
    "Stress / Anxiety",
    "Depression",
    "Obesity / Metabolic syndrome",
    "Alzheimer’s",
    "CKD"
]

class HealthAgent:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.state = AgentState()
        self.model = genai.GenerativeModel(MODEL_NAME)

    def step(self, user_input: str) -> Dict[str, Any]:
        """Main agent loop step."""
        
        # 0. Safety Check (Emergency)
        if self._check_emergency(user_input):
            return {
                "type": "emergency",
                "message": "⚠️ EMERGENCY ALERT: Your symptoms (chest pain, severe breathlessness, or fainting) require IMMEDIATE medical attention. Please call emergency services or go to the nearest hospital right away. I cannot provide further assistance."
            }

        # 1. Update Conversation History
        self.state.conversation_history.append({"role": "user", "content": user_input})
        self.state.turn_count += 1

        # 2. Extract Symptoms & Update State
        self._extract_information(user_input)

        # 3. Calculate Risks
        self._calculate_risks()

        # 4. Check Termination Condition
        top_disease, top_score, second_score = self._get_top_risks()
        
        is_confident = (top_score >= CONFIDENCE_THRESHOLD) and \
                       ((top_score - second_score) >= SCORE_DIFF_THRESHOLD)
        
        if is_confident or self.state.turn_count >= MAX_TURNS:
            self.state.is_finished = True
            return self._generate_final_report(top_disease, top_score)
        
        # 5. Generate Follow-up
        followup = self._generate_followup()
        self.state.conversation_history.append({"role": "assistant", "content": followup["question"]})
        return followup

    def _check_emergency(self, text: str) -> bool:
        """Heuristic check for emergency keywords."""
        emergency_keywords = [
            "chest pain", "crushing chest", "heart attack", 
            "fainted", "unconscious", "severe breathlessness", "can't breathe",
            "coughing blood", "stroke", "sudden blindness"
        ]
        text_lower = text.lower()
        return any(k in text_lower for k in emergency_keywords)

    def _extract_information(self, text: str):
        """Extracts symptoms and lifestyle info using Gemini."""
        prompt = f"""
        Extract medical symptoms, lifestyle data, and family history from the user's text.
        Return a JSON object with keys: 'symptoms' (list of strings), 'lifestyle' (dictionary), 'family_history' (dictionary).
        Only extract explicitly stated information.
        
        User Text: "{text}"
        
        Current State Context:
        Symptoms: {self.state.symptoms}
        History: {self.state.history}
        """
        
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            data = json.loads(response.text)
            
            # Update state
            if "symptoms" in data:
                for s in data["symptoms"]:
                    if s not in self.state.symptoms:
                        self.state.symptoms.append(s)
            
            if "lifestyle" in data:
                self.state.history.update(data["lifestyle"])
                
            if "family_history" in data:
                if "family_history" not in self.state.history:
                    self.state.history["family_history"] = {}
                if isinstance(data["family_history"], dict):
                    self.state.history["family_history"].update(data["family_history"])
                    
        except Exception as e:
            logger.error(f"Error extracting info: {e}")

    def _calculate_risks(self):
        """Hybrid Risk Scoring: Rule-based + LLM Probability."""
        
        # A. Rule-based Scoring (Simplified heuristics)
        rule_scores = {d: 0.0 for d in DISEASES}
        symptoms_str = ", ".join(self.state.symptoms).lower()
        hist = self.state.history
        
        # Diabetes Rules
        if "thirst" in symptoms_str or "urination" in symptoms_str: rule_scores["Diabetes"] += 0.3
        if "sugar" in str(hist): rule_scores["Diabetes"] += 0.2
        
        # Hypertension Rules
        if "headache" in symptoms_str or "dizzy" in symptoms_str: rule_scores["Hypertension"] += 0.2
        if "salt" in str(hist): rule_scores["Hypertension"] += 0.1
        
        # Mental Health Rules
        if "sad" in symptoms_str or "hopeless" in symptoms_str: rule_scores["Depression"] += 0.4
        if "anxious" in symptoms_str or "worry" in symptoms_str: rule_scores["Stress / Anxiety"] += 0.4
        
        # Alzheimer's Rules
        if "forget" in symptoms_str or "memory" in symptoms_str: rule_scores["Alzheimer’s"] += 0.4
        
        # B. LLM Probability Estimation
        llm_scores = self._get_llm_risk_scores()
        
        # C. Combined Confidence
        # final = 0.6 * model + 0.4 * rule
        for d in DISEASES:
            r_score = min(rule_scores.get(d, 0.0), 1.0) # Cap at 1.0
            l_score = llm_scores.get(d, 0.0)
            
            final_conf = (0.6 * l_score) + (0.4 * r_score)
            self.state.risk_scores[d] = round(final_conf, 3)

    def _get_llm_risk_scores(self) -> Dict[str, float]:
        """Asks Gemini to estimate probability for each disease."""
        prompt = f"""
        Analyze the following patient data and estimate the risk probability (0.0 to 1.0) for these diseases:
        {DISEASES}
        
        Patient Data:
        Symptoms: {self.state.symptoms}
        History: {self.state.history}
        
        Return ONLY a JSON object mapping disease names to float probabilities.
        Example: {{"Diabetes": 0.1, "Hypertension": 0.05, ...}}
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error getting LLM scores: {e}")
            return {d: 0.0 for d in DISEASES}

    def _get_top_risks(self):
        """Returns (top_disease, top_score, second_score)."""
        sorted_risks = sorted(self.state.risk_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_risks:
            return None, 0.0, 0.0
        
        top_disease, top_score = sorted_risks[0]
        second_score = sorted_risks[1][1] if len(sorted_risks) > 1 else 0.0
        return top_disease, top_score, second_score

    def _generate_followup(self) -> Dict[str, Any]:
        """Generates the next best question."""
        prompt = f"""
        Based on the current knowledge, generate ONE single follow-up question to clarify the diagnosis.
        Focus on distinguishing between the top likely conditions.
        
        Current Symptoms: {self.state.symptoms}
        Current Risk Scores: {self.state.risk_scores}
        
        Return JSON format:
        {{
          "type": "followup",
          "question": "The question string",
          "feature_target": "The specific symptom/factor you are checking (e.g., 'polydipsia')",
          "options": ["yes", "no", "sometimes"]
        }}
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error generating follow-up: {e}")
            return {
                "type": "followup",
                "question": "Can you describe any other symptoms you are experiencing?",
                "feature_target": "general_symptoms",
                "options": ["open-ended"]
            }

    def _generate_final_report(self, disease: str, probability: float) -> Dict[str, Any]:
        """Generates the final structured report."""
        prompt = f"""
        Generate a final health analysis report for a patient at risk of {disease} (Probability: {probability}).
        
        Patient Data:
        Symptoms: {self.state.symptoms}
        History: {self.state.history}
        
        Requirements:
        1. Symptom Trigger Report: 120 words explaining patterns/triggers.
        2. Plan: 3 specific exercises/yoga/meditation items.
        3. Medicine: OTC-safe only (e.g., ORS, Multivitamins). NO PRESCRIPTION DRUGS (No Metformin, Insulin, BP meds, etc.).
           MUST include "Consult doctor" warning.
        4. Motivation: Supportive message.
        5. Disclaimer: Standard AI disclaimer.
        
        Return JSON matching this schema:
        {{
          "type": "final_report",
          "detected_condition": "{disease}",
          "probability": {probability},
          "symptom_trigger_report": "...",
          "exercise_meditation_yoga_plan": [
            {{"name": "...", "duration_mins": int, "frequency": "..."}}
          ],
          "safe_medicine_info": [
            {{"name": "...", "usage": "...", "notes": "..."}}
          ],
          "motivation": "...",
          "disclaimer": "..."
        }}
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            report = json.loads(response.text)
            # Enforce override of probability and type just in case
            report["type"] = "final_report"
            report["detected_condition"] = disease
            report["probability"] = probability
            return report
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
            return {
                "type": "error",
                "message": "Failed to generate final report. Please consult a doctor."
            }
