import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Requirement 2: Enable CORS so the grader can call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pull the credentials from the environment variables you exported
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN")

# Initialize the OpenAI client pointing to Groq's custom URL proxy
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# Define what the incoming request payload looks like
class QARequest(BaseModel):
    image_base64: str
    question: str

# Requirement 1 & API Spec: POST /answer-image
@app.post("/answer-image")
async def answer_image(payload: QARequest):
    try:
        # Ensure the base64 string includes the correct data URI prefix if not present
        b64_string = payload.image_base64
        if not b64_string.startswith("data:image"):
            b64_string = f"data:image/png;base64,{b64_string}"

        # Ask the Groq vision model (llama-3.3-70b-versatile handles vision)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"{payload.question}\n\nRule: Return ONLY the raw final answer/number text. Do not include units, currency symbols, or extra sentences."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": b64_string
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        # Extract the answer text string
        answer_text = response.choices[0].message.content.strip()
        
        return {"answer": answer_text}
        
    except Exception as e:
        return {"answer": f"Error processing image: {str(e)}"}