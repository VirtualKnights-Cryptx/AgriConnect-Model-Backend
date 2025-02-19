import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

def get_instructions(plant_type, plant_age, disease):
    prompt_healthy = f"""
    A farmer has a {plant_type} plant age around {plant_age}, but it is healthy. Provide a simple step-by-step guide on how to maintain it.
    Use clear and practical steps. First tell your plant is healthy.
    """

    prompt = f"""
    A farmer has a {plant_type} plant age around {plant_age}, affected by {disease} disease. Provide a simple step-by-step guide on how to treat it.
    Use clear and practical steps.
    """
    if disease == "Healthy":
        response = client.models.generate_content(model="gemini-pro", contents=prompt_healthy)
    
    else:   
        response = client.models.generate_content(model="gemini-pro", contents=prompt)

    instructions = response.text if response.text else "No response from Gemini AI."
    
    return instructions
