import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

# Read system instructions
try:
    with open('system_instructions.md', 'r') as f:
        system_instruction = f.read()
except FileNotFoundError:
    raise FileNotFoundError("system_instructions.md not found")

# Initialize model
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-lite',
    system_instruction=system_instruction
)

def generate_report(mri_data, caption):
    """
    Generates a report based on MRI data and image caption using Gemini.
    """
    # Format the input as specified in system instructions
    prompt = f"""
MRI_MODEL_OUTPUT:
Non-Demented: {mri_data['Non-Demented']}
Very Mild Demented: {mri_data['Very Mild Demented']}
Mild Demented: {mri_data['Mild Demented']}
Moderate Demented: {mri_data['Moderate Demented']}

IMAGE_CAPTION:
"{caption}"
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"

if __name__ == "__main__":
    # Dummy Data for Testing
    dummy_mri_data = {
        "Non-Demented": "10%",
        "Very Mild Demented": "85%",
        "Mild Demented": "5%",
        "Moderate Demented": "0%"
    }
    
    dummy_caption = "An MRI scan showing slight hippocampal atrophy consistent with early-stage cognitive decline."
    
    print("Generating Report...\n")
    report = generate_report(dummy_mri_data, dummy_caption)
    print("-" * 50)
    print(report)
    print("-" * 50)
