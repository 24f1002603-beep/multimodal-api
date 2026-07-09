import os
import re
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from google import genai
from google.genai import types

# -----------------------------
# Gemini Client
# -----------------------------
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
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


def decode_image(image_base64: str):

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    return Image.open(BytesIO(image_bytes))


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

        image = decode_image(req.image_base64)

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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                image
            ],
            config=types.GenerateContentConfig(
                temperature=0
            )
        )

        answer = clean_answer(response.text)

        return ImageResponse(answer=answer)

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
