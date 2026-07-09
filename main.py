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
        # Standardize the base64 URI formatting for the vision processor
        if not b64_string.startswith("data:image"):
            b64_string = f"data:image/png;base64,{b64_string}"

        # Using meta-llama/llama-4-scout-17b-16e-instruct for multimodal vision capability
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                f"{payload.question}\n\n"
                                "CRITICAL RULE: Extract the exact value requested from the image. "
                                "Output ONLY the final raw number or string answer. "
                                "Do NOT include any currency symbols ($), units, commas, or explanatory text. "
                                "Example format: 4089.35"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": b64_string}
                        }
                    ]
                }
            ],
            max_tokens=30,
            temperature=0.0  # Force maximum precision and consistency
        )
        
        raw_answer = response.choices[0].message.content.strip()
        
        # --- Post-Processing Data Clean Up ---
        cleaned = raw_answer.strip('"').strip("'").strip()
        
        # Strip currency symbols and formatting commas that violate Rule 1
        if any(char.isdigit() for char in cleaned):
            cleaned = re.sub(r'[$\s€£¥]', '', cleaned)
            if ',' in cleaned and '.' in cleaned:
                cleaned = cleaned.replace(',', '')
            
            # If the model still returned a full phrase, extract just the raw number sequence
            match = re.search(r'\d+\.?\d*', cleaned)
            if match:
                cleaned = match.group(0)

        return {"answer": str(cleaned)}
        
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
