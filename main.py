import os
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN")
)

# Keep the strict response schema required by the specification
class InvoiceResponse(BaseModel):
    invoice_no: Optional[str] = Field(None, description="The invoice number/ID. Null if not found.")
    date: Optional[str] = Field(None, description="The invoice date formatted strictly as YYYY-MM-DD. Null if not found.")
    vendor: Optional[str] = Field(None, description="The name of the vendor/seller. Null if not found.")
    amount: Optional[float] = Field(None, description="The subtotal amount BEFORE tax. Null if not found.")
    tax: Optional[float] = Field(None, description="The tax amount only. Null if not found.")
    currency: Optional[str] = Field(None, description="The 3-letter currency code (e.g., INR, USD). Null if not found.")

@app.post("/extract", response_model=InvoiceResponse)
@app.post("/answer-image", response_model=InvoiceResponse)
async def process_any_invoice(payload: dict[str, Any]):
    """
    Accepts ANY JSON dictionary structure to prevent 422 validation errors, 
    then dynamically extracts the text content to send to the LLM.
    """
    try:
        # 1. Dynamically pull out the text string from the payload regardless of its key
        # Looks for keys like 'invoice_text', 'text', 'image', or falls back to the first available string value.
        invoice_text = ""
        if "invoice_text" in payload:
            invoice_text = str(payload["invoice_text"])
        elif "text" in payload:
            invoice_text = str(payload["text"])
        elif "image" in payload:
            invoice_text = str(payload["image"])
        else:
            # Fallback: look for any string value inside the incoming JSON dictionary
            for value in payload.values():
                if isinstance(value, str) and len(value) > 0:
                    invoice_text = value
                    break
            
            # Absolute fallback: use the string representation of the whole payload
            if not invoice_text:
                invoice_text = str(payload)

        # 2. Instruct the LLM to parse it according to the schema rules
        system_instruction = (
            "You are an expert financial data extractor. Analyze the provided raw invoice data/text "
            "and extract the requested fields. Return your answer strictly as a valid JSON object matching the requested schema.\n"
            "CRITICAL RULES:\n"
            "1. The 'date' field MUST be converted into ISO format (YYYY-MM-DD).\n"
            "2. The 'amount' field MUST be the subtotal BEFORE tax.\n"
            "3. The 'tax' field is only the tax amount.\n"
            "4. Convert names to standard currency codes if applicable (e.g., Rs. or INR becomes 'INR')."
        )

        response = client.beta.chat.completions.parse(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": invoice_text}
            ],
            response_format=InvoiceResponse,
            temperature=0.0,
        )

        return response.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
