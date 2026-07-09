import os
import re
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ["ANTHROPIC_AUTH_TOKEN"]
BASE_URL = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")


class ImageRequest(BaseModel):
    image_base64: str
    question: str


class ImageResponse(BaseModel):
    answer: str


def clean_answer(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()

    text = re.sub(r"^```.*?\n", "", text, flags=re.DOTALL)
    text = text.replace("```", "")

    return text.strip("\"' ").strip()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/answer-image", response_model=ImageResponse)
def answer(req: ImageRequest):

    image = req.image_base64.strip()

    if not image.startswith("data:image"):
        image = "data:image/png;base64," + image

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "temperature": 0,
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Answer ONLY the user's question.

Question:
{req.question}

Rules:
- Return only the answer.
- No explanation.
- If numeric, return only the number.
- No currency symbols.
- No units.
- No markdown.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )

    # This is the important part.
    # If Groq rejects the request you'll actually see WHY.
    if r.status_code != 200:
        print(r.status_code)
        print(r.text)
        raise HTTPException(r.status_code, r.text)

    data = r.json()

    answer = data["choices"][0]["message"]["content"]

    return ImageResponse(answer=clean_answer(answer))
