import os
import re
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# -----------------------------
# AI Pipe Client (OpenAI Compatible)
# -----------------------------
client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Multimodal Image QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class ImageRequest(BaseModel):
    image_base64: str
    question: str


class ImageResponse(BaseModel):
    answer: str


# -----------------------------
# Helpers
# -----------------------------
def clean_answer(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()

    text = re.sub(r"^```.*?\n", "", text, flags=re.DOTALL)
    text = text.replace("```", "")
    text = text.strip()

    if (
        text.startswith('"')
        and text.endswith('"')
        and len(text) >= 2
    ):
        text = text[1:-1]

    return text.strip()


def format_base64_image(image_base64: str) -> str:
    # Strip any existing data URI header prefix so we can standardize it
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    return f"data:image/jpeg;base64,{image_base64.strip()}"


# -----------------------------
# Root
# -----------------------------
@app.get("/")
def home():
    return {"status": "running"}


# -----------------------------
# Main Endpoint
# -----------------------------
@app.post("/answer-image", response_model=ImageResponse)
def answer_image(req: ImageRequest):

    try:
        # Prepare the standard base64 data URL for OpenAI/AI Pipe integration
        image_url = format_base64_image(req.image_base64)

        prompt = f"""
You are an expert OCR and visual reasoning model.

Answer the user's question using ONLY the image.

Question:
{req.question}

IMPORTANT RULES

1. Return ONLY the answer.
2. No explanation.
3. No markdown.
4. No code block.
5. No extra words.
6. If numeric, return only the number.
7. Remove currency symbols.
8. Remove units.
9. Keep dates exactly as shown unless asked otherwise.
"""

        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )

        answer = clean_answer(response.choices[0].message.content)

        return ImageResponse(answer=answer)

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
