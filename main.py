import os
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Multimodal Image QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url=os.environ["ANTHROPIC_BASE_URL"],
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
)


class ImageRequest(BaseModel):
    image_base64: str
    question: str


class ImageResponse(BaseModel):
    answer: str


def clean_answer(ans: str) -> str:
    ans = ans.strip()

    # remove markdown code fences
    ans = re.sub(r"^```.*?\n", "", ans, flags=re.DOTALL)
    ans = ans.replace("```", "")

    # remove surrounding quotes
    ans = ans.strip("\"'")
    return ans.strip()


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/answer-image", response_model=ImageResponse)
def answer_image(req: ImageRequest):

    try:

        image_data = req.image_base64.strip()

        # add prefix if grader sends only raw base64
        if not image_data.startswith("data:image"):
            image_data = "data:image/png;base64," + image_data

        prompt = f"""
You are an OCR and visual reasoning assistant.

Answer ONLY the user's question.

Question:
{req.question}

Rules:

- Return ONLY the answer.
- No explanation.
- No sentences.
- If numeric, return ONLY the number.
- No commas unless present in the answer.
- No currency symbols.
- No units.
- No markdown.
"""

        completion = client.chat.completions.create(

            model="meta-llama/llama-4-scout-17b-16e-instruct",

            temperature=0,

            max_completion_tokens=128,

            messages=[
                {
                    "role": "user",
                    "content": [

                        {
                            "type": "text",
                            "text": prompt,
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            },
                        },
                    ],
                }
            ],
        )

        answer = completion.choices[0].message.content

        if answer is None:
            answer = ""

        answer = clean_answer(answer)

        return ImageResponse(answer=answer)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
