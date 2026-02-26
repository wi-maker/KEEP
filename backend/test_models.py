import os
from dotenv import load_dotenv
import google.generativeai as genai
import warnings

warnings.filterwarnings('ignore')
load_dotenv('.env')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)
