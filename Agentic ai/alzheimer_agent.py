import os
import json
import logging
import math
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
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
    logger.warning("GOOGLE_API_KEY not found. Please set it.")

genai.configure(api_key=GOOGLE_API_KEY)

# Constants
MODEL_NAME = "gemini-2.0-flash"
FALLBACK_MODEL_NAME = "gemini-2.0-flash-lite"
MAX_ROUNDS = 5
CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.12
W0 = -2.2 # Baseline offset

# Feature Definitions & Normalization
FEATURES = {
    "forget_recent": {"type": "categorical", "values": {"yes": 1.0, "sometimes": 0.5, "no": 0.0}},
    "daily_task_difficulty": {"type": "categorical", "values": {"major": 1.0, "mild": 0.5, "no": 0.0}},
    "confused_time_place": {"type": "categorical", "values": {"yes": 1.0, "no": 0.0}},
    "family_history": {"type": "categorical", "values": {"1st_degree": 1.0, "none": 0.0}},
    "duration_months": {"type": "numeric", "max": 24.0}, 
    "diabetes": {"type": "categorical", "values": {"yes": 1.0, "unsure": 0.5, "no": 0.0}},
    "hypertension": {"type": "categorical", "values": {"yes": 1.0, "unsure": 0.5, "no": 0.0}},
    "disorientation_freq": {"type": "categorical", "values": {"often": 1.0, "sometimes": 0.5, "never": 0.0}},
    "sleep_hours": {"type": "numeric", "max": 12.0},
    "mood_low": {"type": "categorical", "values": {"yes": 1.0, "no": 0.0}},
    "weight_change": {"type": "categorical", "values": {"loss": 1.0, "gain": 0.5, "none": 0.0}}
}

class AgentSession(BaseModel):
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    features: Dict[str, Any] = Field(default_factory=dict) # Raw values
    normalized_features: Dict[str, float] = Field(default_factory=dict) # Numeric 0-1
    questions_asked: List[Dict[str, Any]] = Field(default_factory=list)
    rounds: int = 0
    is_finished: bool = False
    final_report: Optional[Dict[str, Any]] = None
    last_question_target: Optional[str] = None

class AlzheimerAgent:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_dir = os.path.join(os.path.dirname(__file__), "sessions")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Load session if exists
        self.session = self._load_session(self.session_id)
        
        self.kb = self._load_kb()
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.fallback_model = genai.GenerativeModel(FALLBACK_MODEL_NAME)

    def _load_kb(self):
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        kb = {}
        try:
            with open(os.path.join(kb_path, "exercises.json"), "r") as f: kb["exercises"] = json.load(f)
            with open(os.path.join(kb_path, "supplements.json"), "r") as f: kb["supplements"] = json.load(f)
            with open(os.path.join(kb_path, "triggers.json"), "r") as f: kb["triggers"] = json.load(f)
            with open(os.path.join(kb_path, "meditation.json"), "r") as f: kb["meditation"] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading KB: {e}")
            kb = {"exercises": [], "supplements": [], "triggers": {}, "meditation": []}
        return kb

    def _load_session(self, session_id: str) -> AgentSession:
        path = os.path.join(self.session_dir, f"{session_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    return AgentSession(**data)
            except Exception as e:
                logger.error(f"Error loading session: {e}")
        return AgentSession(session_id=session_id)

    def _save_session(self):
        path = os.path.join(self.session_dir, f"{self.session_id}.json")
        try:
            with open(path, "w") as f:
                f.write(self.session.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def _generate_content_safe(self, prompt: str) -> Optional[str]:
        """Generates content with fallback logic."""
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return response.text
        except Exception as e:
            logger.warning(f"Primary model {MODEL_NAME} failed: {e}. Trying fallback {FALLBACK_MODEL_NAME}...")
            try:
                response = self.fallback_model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                return response.text
            except Exception as e2:
                logger.error(f"Fallback model {FALLBACK_MODEL_NAME} also failed: {e2}")
                return None

    def step(self, user_text: str) -> Dict[str, Any]:
        """Main agent loop."""
        
        # 1. Emergency Check
        if self._check_red_flags(user_text):
            resp = {
                "type": "emergency",
                "message": "If you are experiencing chest pain, fainting, severe shortness of breath, call emergency services immediately.",
                "recommended_action": "Call local emergency number or visit nearest ER."
            }
            self.session.is_finished = True
            self._save_session()
            return resp

        # 2. Extract & Update Features
        if user_text:
            self._extract_features(user_text)
            # Log Q&A
            if self.session.last_question_target:
                 self.session.questions_asked.append({
                     "q": self.session.last_question_target, # Storing target as Q identifier for now
                     "answer": user_text,
                     "ts": datetime.now().isoformat()
                 })

        # 3. Check Decision Logic
        p_rule = self._compute_rule_score()
        p_llm = self._get_llm_probability()
        
        final_conf = (0.6 * p_llm) + (0.4 * p_rule)
        
        logger.info(f"Scores - Rule: {p_rule:.2f}, LLM: {p_llm:.2f}, Final: {final_conf:.2f}")

        # Decision Rule
        # Assume second_best is low (~0.1) for now as we only track Alzheimer's
        second_best = 0.1 
        
        if (final_conf >= CONFIDENCE_THRESHOLD and (final_conf - second_best) >= MARGIN_THRESHOLD) or self.session.rounds >= MAX_ROUNDS:
            # Generate Report
            self.session.is_finished = True
            report = self._generate_final_report(final_conf)
            self.session.final_report = report
            self._save_session()
            return report
        
        # 4. Generate Next Question
        next_q = self._generate_next_question()
        self.session.last_question_target = next_q.get("feature_target")
        self._save_session()
        return next_q

    def _check_red_flags(self, text: str) -> bool:
        keywords = ["chest pain", "fainted", "unconscious", "can't breathe", "severe confusion", "sudden blindness", "shortness of breath"]
        return any(k in text.lower() for k in keywords)

    def _extract_features(self, text: str):
        prompt = f"""
        Extract Alzheimer's risk features from the text.
        Map to these keys and values:
        {json.dumps(FEATURES, indent=2)}
        
        User Text: "{text}"
        Current Features: {json.dumps(self.session.features)}
        
        Return JSON object with keys and RAW values.
        """
        try:
            response_text = self._generate_content_safe(prompt)
            if not response_text:
                raise Exception("LLM generation failed")
                
            data = json.loads(response_text)
            
            for k, v in data.items():
                if k in FEATURES:
                    self.session.features[k] = v
                    self.session.normalized_features[k] = self._normalize_feature(k, v)
            
            # Fallback for last target
            if self.session.last_question_target and self.session.last_question_target not in data:
                text_lower = text.lower()
                target = self.session.last_question_target
                if "yes" in text_lower:
                    self.session.features[target] = "yes"
                    self.session.normalized_features[target] = self._normalize_feature(target, "yes")
                elif "no" in text_lower:
                    self.session.features[target] = "no"
                    self.session.normalized_features[target] = self._normalize_feature(target, "no")
                else:
                    self.session.features[target] = "unknown"
                    self.session.normalized_features[target] = 0.0
                    
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            if self.session.last_question_target:
                self.session.features[self.session.last_question_target] = "error_skipped"
                self.session.normalized_features[self.session.last_question_target] = 0.0

    def _normalize_feature(self, key: str, value: Any) -> float:
        f_def = FEATURES[key]
        if f_def["type"] == "categorical":
            if isinstance(value, str): value = value.lower()
            return f_def["values"].get(value, 0.0)
        elif f_def["type"] == "numeric":
            try:
                # Extract number from string if needed
                if isinstance(value, str):
                    import re
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", value)
                    if nums: val = float(nums[0])
                    else: return 0.0
                else:
                    val = float(value)
                return min(val / f_def["max"], 1.0)
            except:
                return 0.0
        return 0.0

    def _compute_rule_score(self) -> float:
        nf = self.session.normalized_features
        
        forget = nf.get("forget_recent", 0)
        task = nf.get("daily_task_difficulty", 0)
        confused = nf.get("confused_time_place", 0)
        family = nf.get("family_history", 0)
        
        # Duration raw months
        raw_dur = 0
        if "duration_months" in self.session.features:
            try: 
                val = self.session.features["duration_months"]
                if isinstance(val, str):
                    import re
                    nums = re.findall(r"\d+", val)
                    if nums: raw_dur = float(nums[0])
                else:
                    raw_dur = float(val)
            except: pass
            
        diabetes = nf.get("diabetes", 0)
        htn = nf.get("hypertension", 0)
        disorient = nf.get("disorientation_freq", 0)
        
        sleep_hours = 8
        if "sleep_hours" in self.session.features:
             # Try to get raw sleep hours
             try:
                 val = self.session.features["sleep_hours"]
                 if isinstance(val, str):
                    import re
                    nums = re.findall(r"\d+", val)
                    if nums: sleep_hours = float(nums[0])
                 else:
                     sleep_hours = float(val)
             except: pass
        
        logit = (W0 
                 + 0.25 * forget 
                 + 0.20 * task
                 + 0.18 * confused
                 + 0.15 * family
                 + 0.10 * (raw_dur / 12.0)
                 + 0.05 * diabetes
                 + 0.05 * htn
                 + 0.07 * disorient
                 - 0.02 * (0.1 if sleep_hours < 6 else 0))
        
        return 1.0 / (1.0 + math.exp(-logit))

    def _get_llm_probability(self) -> float:
        prompt = f"""
        SYSTEM: You are a clinical-risk estimator (not a doctor). Output ONLY a JSON object.
        INPUT: {{ "features": {json.dumps(self.session.features)} }}
        TASK: Given features, estimate probability that user is showing early Alzheimer’s / significant cognitive decline.
        OUTPUT FORMAT:
        {{"condition":"AlzheimerRisk","likelihood":0.0, "explain":"one-line rationale (max 20 words)"}}
        Constraints:
        - Provide value 0..1 rounded to two decimals.
        - Tone: clinical-neutral (this is internal).
        - No medical instructions here.
        """
        try:
            response_text = self._generate_content_safe(prompt)
            if not response_text:
                return 0.0
            return json.loads(response_text).get("likelihood", 0.0)
        except:
            return 0.0

    def _generate_next_question(self) -> Dict[str, Any]:
        priority = [
            ("forget_recent", "Do you forget recent conversations often?", ["yes", "no", "sometimes"]),
            ("duration_months", "How long have these memory lapses lasted?", ["months", "years"]),
            ("daily_task_difficulty", "Are you having trouble with daily tasks?", ["yes", "no"]),
            ("confused_time_place", "Do you feel confused about dates or places?", ["yes", "no"]),
            ("family_history", "Any family history of dementia?", ["yes", "no"]),
            ("sleep_hours", "How many hours do you sleep nightly?", ["numeric"]),
            ("diabetes", "Do you have diabetes?", ["yes", "no"]),
            ("hypertension", "Do you have high blood pressure?", ["yes", "no"])
        ]
        
        for key, q_text, opts in priority:
            if key not in self.session.features:
                self.session.rounds += 1
                return {
                    "type": "followup",
                    "question": q_text,
                    "feature_target": key,
                    "expected_answers": opts,
                    "round": self.session.rounds,
                    "session_id": self.session_id
                }
        
        # Force finish if all asked
        self.session.rounds = MAX_ROUNDS
        return self.step("") 

    def _generate_final_report(self, probability: float) -> Dict[str, Any]:
        risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"
        
        prompt = f"""
        SYSTEM: You are a friendly, casual health assistant providing structured educational output. NOT a doctor. Use provided KB for exercises and safe supplements.
        INPUT: {{ "features": {json.dumps(self.session.features)}, "kb_snippets": {json.dumps(self.kb)} }}
        TASK: Produce EXACT JSON with fields:
        {{ 
          "type": "final_report",
          "detected_condition": "AlzheimerRisk", 
          "risk_level": "{risk_level}", 
          "probability": {probability:.2f}, 
          "symptom_trigger_report": "short paragraph <= 140 words", 
          "exercise_meditation_yoga_plan": [{{ "name": "...", "duration_mins": int, "frequency": "..." }}], 
          "safe_medicine_info": [{{ "name": "...", "short_note": "..." }}], 
          "motivation": "Friendly 1-2 sentence tailored message.", 
          "disclaimer": "I am an AI assistant, not a doctor. Consult a specialist for diagnosis and treatment.",
          "session_summary": {{ "features": {json.dumps(self.session.normalized_features)}, "questions_asked": {json.dumps(self.session.questions_asked)}, "rounds": {self.session.rounds} }},
          "session_id": "{self.session_id}"
        }}
        Constraints:
        - report <= 140 words
        - exercise items: Indian-friendly names allowed
        - safe_medicine_info: short by default; include dosage only if input asks for "dosage"
        - Always append disclaimer.
        - Output ONLY valid JSON.
        Tone: friendly, casual.
        """
        try:
            response_text = self._generate_content_safe(prompt)
            if not response_text:
                raise Exception("Report generation failed")
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Report gen error: {e}")
            return {"type": "error", "message": "Report generation failed."}
