import os
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

class QARequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: QARequest):
    try:
        b64_string = payload.image_base64
        if not b64_string.startswith("data:image"):
            b64_string = f"data:image/png;base64,{b64_string}"

        # Using a strict system prompt to force the model to comply
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict data extraction tool. Analyze the image and answer the question. "
                        "CRITICAL: Output ONLY the final raw answer string. Do not include currency symbols ($), "
                        "do not include units, do not include commas in numbers, and do not write complete sentences. "
                        "Example: If the answer is $4,089.35, output exactly: 4089.35"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": payload.question},
                        {
                            "type": "image_url",
                            "image_url": {"url": b64_string}
                        }
                    ]
                }
            ],
            max_tokens=50,
            temperature=0.0  # Make the model completely deterministic
        )
        
        raw_answer = response.choices[0].message.content.strip()
        
        # --- Python Post-Processing Cleaning ---
        # Remove any leading/trailing quotes the model might add
        cleaned = raw_answer.strip('"').strip("'")
        
        # If the answer looks like it contains a numeric value, let's clean it explicitly
        # This removes currency symbols like $, €, £ and strips out thousands-separator commas
        if any(char.isdigit() for char in cleaned):
            # Strip common currency signs
            cleaned = re.sub(r'[$\s€£¥]', '', cleaned)
            # Remove commas only if they act as a thousands separator (e.g., 4,089.35 -> 4089.35)
            if ',' in cleaned and '.' in cleaned:
                cleaned = cleaned.replace(',', '')
            # If the model still wrote a whole sentence, try to grab just the number pattern
            match = re.search(r'\d+\.?\d*', cleaned)
            if match:
                cleaned = match.group(0)

        return {"answer": str(cleaned)}
        
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
