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
MODEL_NAME = "gemini-2.0-flash-lite" # Specific version
FALLBACK_MODEL_NAME = "gemini-pro"
MAX_ROUNDS = 5
CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.12

# Feature Definitions & Normalization
FEATURES = {
    "forget_recent": {"type": "categorical", "values": {"yes": 1.0, "sometimes": 0.5, "no": 0.0}},
    "daily_task_difficulty": {"type": "categorical", "values": {"major": 1.0, "mild": 0.5, "no": 0.0}},
    "confused_time_place": {"type": "categorical", "values": {"yes": 1.0, "no": 0.0}},
    "family_history": {"type": "categorical", "values": {"1st_degree": 1.0, "none": 0.0}},
    "duration_months": {"type": "numeric", "max": 24.0}, # Cap at 24 months for normalization
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
        self.session = AgentSession(session_id=self.session_id)
        self.kb = self._load_kb()
        self.model = self._init_model()

    def _init_model(self):
        try:
            return genai.GenerativeModel(MODEL_NAME)
        except:
            logger.warning(f"Failed to init {MODEL_NAME}, trying fallback {FALLBACK_MODEL_NAME}")
            return genai.GenerativeModel(FALLBACK_MODEL_NAME)

    def _load_kb(self):
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
        kb = {}
        try:
            with open(os.path.join(kb_path, "exercises.json"), "r") as f: kb["exercises"] = json.load(f)
            with open(os.path.join(kb_path, "medicines.json"), "r") as f: kb["medicines"] = json.load(f)
            with open(os.path.join(kb_path, "triggers.json"), "r") as f: kb["triggers"] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading KB: {e}")
            kb = {"exercises": [], "medicines": [], "triggers": {}}
        return kb

    def step(self, user_text: str) -> Dict[str, Any]:
        """Main agent loop."""

        # 1. Emergency Check
        if self._check_red_flags(user_text):
            return {
                "type": "emergency",
                "message": "⚠️ EMERGENCY: Your symptoms (chest pain, fainting, severe confusion, or breathing difficulty) require IMMEDIATE medical attention. Please call emergency services."
            }

        # 2. Extract & Update Features
        self._extract_features(user_text)

        # 3. Check Decision Logic
        p_rule = self._compute_rule_score()
        p_llm = self._get_llm_probability()

        final_conf = (0.6 * p_llm) + (0.4 * p_rule)

        logger.info(f"Scores - Rule: {p_rule:.2f}, LLM: {p_llm:.2f}, Final: {final_conf:.2f}")

        if (final_conf >= CONFIDENCE_THRESHOLD and (final_conf - 0.5) >= MARGIN_THRESHOLD) or self.session.rounds >= MAX_ROUNDS:
            # Generate Report
            self.session.is_finished = True
            return self._generate_final_report(final_conf)

        # 4. Generate Next Question
        next_q = self._generate_next_question()
        self.session.last_question_target = next_q.get("feature_target")
        return next_q

    def _check_red_flags(self, text: str) -> bool:
        keywords = ["chest pain", "fainted", "unconscious", "can't breathe", "severe confusion", "sudden blindness"]
        return any(k in text.lower() for k in keywords)

    def _extract_features(self, text: str):
        """Uses Gemini to extract features from text."""
        prompt = f"""
        Extract Alzheimer's risk features from the text.
        Map to these keys and values:
        {json.dumps(FEATURES, indent=2)}

        User Text: "{text}"
        Current Features: {json.dumps(self.session.features)}

        Return JSON object with keys and RAW values (e.g., "yes", "3 months").
        Only include features explicitly mentioned.
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            data = json.loads(response.text)

            for k, v in data.items():
                if k in FEATURES:
                    self.session.features[k] = v
                    self.session.normalized_features[k] = self._normalize_feature(k, v)

            # If tracking a specific target and it wasn't found, try simple heuristic or mark skipped
            if self.session.last_question_target and self.session.last_question_target not in data:
                # Fallback: if user said "yes" or "no", map it
                text_lower = text.lower()
                target = self.session.last_question_target
                if "yes" in text_lower:
                    self.session.features[target] = "yes"
                    self.session.normalized_features[target] = self._normalize_feature(target, "yes")
                elif "no" in text_lower:
                    self.session.features[target] = "no"
                    self.session.normalized_features[target] = self._normalize_feature(target, "no")
                else:
                    # Mark as skipped/unknown to prevent loop
                    self.session.features[target] = "unknown"
                    self.session.normalized_features[target] = 0.0

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # Failsafe: Mark last target as unknown to proceed
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
                val = float(value)
                return min(val / f_def["max"], 1.0)
            except:
                return 0.0
        return 0.0

    def _compute_rule_score(self) -> float:
        """
        p_rule = sigmoid( w0 + ... )
        """
        nf = self.session.normalized_features

        # Default to 0 if not present
        forget = nf.get("forget_recent", 0)
        task = nf.get("daily_task_difficulty", 0)
        confused = nf.get("confused_time_place", 0)
        family = nf.get("family_history", 0)
        duration = nf.get("duration_months", 0) # Already normalized /24, but formula says /12.
        # Let's stick to the spec formula: 0.10 * (months/12)
        # We need raw months for this specific term if we want to follow spec EXACTLY.
        # Or we can just use the normalized value if we adjust the weight.
        # Let's use raw value if available.
        raw_dur = 0
        if "duration_months" in self.session.features:
            try: raw_dur = float(self.session.features["duration_months"])
            except: pass

        diabetes = nf.get("diabetes", 0)
        htn = nf.get("hypertension", 0)
        disorient = nf.get("disorientation_freq", 0)

        sleep_hours = 8 # Default
        if "sleep_hours" in self.session.features:
            try: sleep_hours = float(self.session.features["sleep_hours"])
            except: pass

        # w0 baseline offset. Let's pick -3.0 so that 0 features -> low prob.
        w0 = -3.0

        logit = (w0
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
        Estimate probability (0.0 to 1.0) of early Alzheimer's / cognitive decline.
        Features: {json.dumps(self.session.features)}
        Return JSON: {{"likelihood": 0.5}}
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text).get("likelihood", 0.0)
        except:
            return 0.0

    def _generate_next_question(self) -> Dict[str, Any]:
        """Selects next question based on missing high-value features."""
        # Priority order of features to ask
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

        # If all asked, force finish
        self.session.rounds = MAX_ROUNDS
        return self.step("") # Recursively call to trigger finish

    def _generate_final_report(self, probability: float) -> Dict[str, Any]:
        risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"

        prompt = f"""
        Generate final Alzheimer's report.
        Risk: {risk_level} (Prob: {probability})
        Features: {json.dumps(self.session.features)}
        KB: {json.dumps(self.kb)}

        Return JSON:
        {{
          "type": "final_report",
          "detected_condition": "AlzheimerRisk",
          "risk_level": "{risk_level}",
          "probability": {probability},
          "symptom_trigger_report": "...",
          "exercise_meditation_yoga_plan": [...],
          "safe_medicine_info": [...],
          "motivation": "...",
          "disclaimer": "..."
        }}
        """
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Report gen error: {e}")
            return {"type": "error", "message": "Report generation failed."}
