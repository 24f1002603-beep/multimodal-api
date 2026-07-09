import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI  # Switched to OpenAI client to match the /v1 endpoint layout

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force the client to read your specific dashboard environment variables
client = OpenAI(
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN")
)

class InvoiceRequest(BaseModel):
    invoice_text: str

class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = Field(None, description="The invoice number/ID. Null if not found.")
    date: Optional[str] = Field(None, description="The invoice date formatted strictly as YYYY-MM-DD. Null if not found.")
    vendor: Optional[str] = Field(None, description="The name of the vendor/seller. Null if not found.")
    amount: Optional[float] = Field(None, description="The subtotal amount BEFORE tax. Null if not found.")
    tax: Optional[float] = Field(None, description="The tax amount only. Null if not found.")
    currency: Optional[str] = Field(None, description="The 3-letter currency code (e.g., INR, USD). Null if not found.")

@app.post("/extract")
async def extract_invoice(payload: InvoiceRequest):
    try:
        system_instruction = (
            "You are an expert financial data extractor. Analyze the provided raw invoice text "
            "and extract the requested fields. Return your answer strictly as a valid JSON object matching the requested schema.\n"
            "CRITICAL RULES:\n"
            "1. The 'date' field MUST be converted into ISO format (YYYY-MM-DD).\n"
            "2. The 'amount' field MUST be the subtotal BEFORE tax.\n"
            "3. The 'tax' field is only the tax amount.\n"
            "4. Convert names to standard currency codes if applicable (e.g., Rs. or INR becomes 'INR')."
        )

        # Using structured responses via standard chat completions 
        response = client.beta.chat.completions.parse(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": payload.invoice_text}
            ],
            response_format=InvoiceResponse,
            temperature=0.0,
        )

        return response.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
