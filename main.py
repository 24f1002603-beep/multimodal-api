import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Enable CORS (Crucial so the IITM Cloudflare Worker grader doesn't block your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 3. Initialize the Groq Client
# It automatically picks up the GROQ_API_KEY environment variable
client = Groq()

# 4. Define the exact request format required by the API specification
class QAQuery(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(query: QAQuery):
    try:
        # Standardizing the base64 string format for the API data URL
        base64_image = query.image_base64
        if not base64_image.startswith("data:image"):
            # Assuming it's a PNG based on the assignment description, but adjusting if needed
            base64_image = f"data:image/png;base64,{base64_image}"

        # 5. Call Groq's Vision Model
        # We prompt the model to ONLY return the raw answer to avoid grading errors
        system_prompt = (
            "You are a precise data extraction assistant. Analyze the image and answer the user's question. "
            "CRITICAL: If the answer is a number, return ONLY the raw numeric value (e.g., '4089.35'). "
            "Do not include currency symbols ($ or ₹), units, spaces, punctuation, or conversational filler like 'The total is...'. "
            "If it is a text answer, keep it brief and precise."
        )

        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview", # High-speed multimodal model on Groq
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_prompt}\n\nQuestion: {query.question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ],
            temperature=0.0, # Keeps the model factual and deterministic
            max_tokens=100
        )

        # 6. Extract raw text response
        raw_answer = response.choices[0].message.content.strip()

        # 7. Post-processing safety layer to prevent "image answers are incorrect" error
        # Strips out common stray characters like currency symbols just in case
        cleaned_answer = raw_answer.replace("$", "").replace("¥", "").replace("€", "").replace("₹", "").strip()

        return {"answer": str(cleaned_answer)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
