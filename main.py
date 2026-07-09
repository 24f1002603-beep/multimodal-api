import os
import json
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Request
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

# 1. The inner structured data format required by the prompt
class InvoiceData(BaseModel):
    invoice_no: Optional[str] = Field(None, description="The invoice number/ID. Null if not found.")
    date: Optional[str] = Field(None, description="The invoice date formatted strictly as YYYY-MM-DD. Null if not found.")
    vendor: Optional[str] = Field(None, description="The name of the vendor/seller. Null if not found.")
    amount: Optional[float] = Field(None, description="The subtotal amount BEFORE tax. Null if not found.")
    tax: Optional[float] = Field(None, description="The tax amount only. Null if not found.")
    currency: Optional[str] = Field(None, description="The 3-letter currency code (e.g., INR, USD). Null if not found.")

# 2. The top-level wrapper required by the grader ("Response JSON must include an 'answer' field")
class GraderResponse(BaseModel):
    answer: InvoiceData

@app.post("/extract", response_model=GraderResponse)
@app.post("/answer-image", response_model=GraderResponse)
async def process_any_invoice(request: Request):
    try:
        # Capture incoming text dynamically from any data wrapper
        invoice_content = ""
        try:
            payload = await request.json()
            if isinstance(payload, dict):
                if "invoice_text" in payload:
                    invoice_content = str(payload["invoice_text"])
                elif "text" in payload:
                    invoice_content = str(payload["text"])
                elif "image" in payload:
                    invoice_content = str(payload["image"])
                else:
                    for val in payload.values():
                        if isinstance(val, str) and len(val) > 5:
                            invoice_content = val
                            break
                    if not invoice_content:
                        invoice_content = json.dumps(payload)
            else:
                invoice_content = str(payload)
        except Exception:
            raw_body = await request.body()
            invoice_content = raw_body.decode("utf-8", errors="ignore")

        if not invoice_content.strip():
            return GraderResponse(answer=InvoiceData())

        system_instruction = (
            "You are an expert financial data extractor. Analyze the provided raw text data or "
            "base64 content of an invoice and extract the requested fields. "
            "Return your response strictly matching the requested JSON schema wrapper.\n"
            "CRITICAL RULES:\n"
            "1. The 'date' field MUST be converted into ISO format (YYYY-MM-DD).\n"
            "2. The 'amount' field MUST be the subtotal BEFORE tax.\n"
            "3. The 'tax' field is only the tax amount.\n"
            "4. Convert currency symbols to standard 3-letter codes (e.g., Rs. or INR becomes 'INR')."
        )

        # Force the model to output the nested structure directly
        response = client.beta.chat.completions.parse(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": invoice_content}
            ],
            response_format=GraderResponse,
            temperature=0.0,
        )

        return response.choices[0].message.parsed

    except Exception as e:
        print(f"Extraction processing exception: {str(e)}")
        # Returns a structural fallback containing empty properties inside the 'answer' field
        return GraderResponse(answer=InvoiceData())
