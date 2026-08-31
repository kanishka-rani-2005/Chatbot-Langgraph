import os
from dotenv import load_dotenv

load_dotenv()

print("GOOGLE_API_KEY exists:", bool(os.getenv("GEMINI_API_KEY")))