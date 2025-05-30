from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import asyncio
from typing import Dict, List, Any
import os
import re
import requests
import pdfkit
import google.generativeai as genai

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-pro')

# Load KYC rules
def load_rules():
    with open("kyc_rules.json", "r") as file:
        return json.load(file)

RULES = load_rules()

# Models
class RiskAssessment(BaseModel):
    risk_score: float
    recommendation: str

class FraudDetection(BaseModel):
    fraud_detected: bool
    fraud_reasons: List[str]

class ComplianceCheck(BaseModel):
    compliance_flags: List[str]
    compliance_issues: List[str]

class KYCRequest(BaseModel):
    data: Dict[str, Any]

class KYCResponse(BaseModel):
    risk: Dict[str, Any]
    fraud: Dict[str, Any]
    compliance: Dict[str, Any]

# Gemini call
def call_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(prompt)
        raw_response = response.text

        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            clean_json = re.sub(r'//.*?\n', '', clean_json)
            clean_json = re.sub(r'/\*.*?\*/', '', clean_json, flags=re.DOTALL)
            return clean_json

        return raw_response
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

# Analysis Functions
async def check_risk(kyc_json: dict):
    prompt = (
        f"You are an expert in risk assessment. Analyze the risk for the following KYC data:\n\n"
        f"{json.dumps(kyc_json, indent=2)}\n\n"
        "This data may contain any information relevant to a customer's financial background, identity documents, "
        "or a combination of both. Consider all available data to assess risk.\n\n"
        "Return your response in valid JSON format with the following structure:\n"
        "{\n"
        "  \"risk_score\": 0.65,\n"
        "  \"recommendation\": \"Your recommendation here\"\n"
        "}"
    )

    response = call_gemini(prompt)

    try:
        parsed_response = json.loads(response)
        return {
            "risk_score": float(parsed_response.get("risk_score", 0.5)),
            "recommendation": str(parsed_response.get("recommendation", "No recommendation provided"))
        }
    except json.JSONDecodeError:
        return {
            "risk_score": 0.5,
            "recommendation": f"Failed to parse response: {response}"
        }

async def check_fraud(kyc_json: dict):
    prompt = (
        f"You are an expert in fraud detection. Analyze the following KYC data for fraud risk:\n\n"
        f"{json.dumps(kyc_json, indent=2)}\n\n"
        "This data may include identity documents, financial statements, or both. Identify any inconsistencies, "
        "anomalies, or potential red flags that may indicate fraudulent activity.\n\n"
        "Return your response in valid JSON format with the following structure:\n"
        "{\n"
        "  \"fraud_detected\": false,\n"
        "  \"fraud_reasons\": []\n"
        "}"
    )

    response = call_gemini(prompt)

    try:
        parsed_response = json.loads(response)
        return {
            "fraud_detected": bool(parsed_response.get("fraud_detected", False)),
            "fraud_reasons": list(parsed_response.get("fraud_reasons", []))
        }
    except json.JSONDecodeError:
        return {
            "fraud_detected": False,
            "fraud_reasons": [f"Failed to parse response: {response}"]
        }

async def check_compliance(kyc_json: dict):
    missing_fields = []
    required_fields = ["nationality", "document_type", "date_of_birth"]

    for field in required_fields:
        if field not in kyc_json or not kyc_json[field]:
            missing_fields.append(field)

    prompt = (
        f"You are a compliance officer. Check the following KYC data for regulatory compliance:\n\n"
        f"{json.dumps(kyc_json, indent=2)}\n\n"
        "This data may contain any combination of personal identification details and financial records. "
        "Identify any missing, incorrect, or non-compliant information.\n\n"
        "Return your response in valid JSON format:\n"
        "{\n"
        "  \"compliance_flags\": [\"List of minor compliance warnings\"],\n"
        "  \"compliance_issues\": [\"List of critical compliance violations\"]\n"
        "}"
    )

    response = call_gemini(prompt)

    try:
        parsed_response = json.loads(response)
        if "compliance_flags" in parsed_response and "compliance_issues" in parsed_response:
            if missing_fields:
                parsed_response["compliance_flags"].append(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )
            return parsed_response
    except json.JSONDecodeError:
        pass

    return {
        "compliance_flags": ["Could not determine compliance status."],
        "compliance_issues": []
    }

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "KYC Analysis API using Gemini is running"}

@app.post("/analyze", response_model=KYCResponse)
async def analyze_kyc(request: KYCRequest):
    try:
        kyc_json = request.data
        risk_result, fraud_result, compliance_result = await asyncio.gather(
            check_risk(kyc_json),
            check_fraud(kyc_json),
            check_compliance(kyc_json)
        )
        return {
            "risk": risk_result,
            "fraud": fraud_result,
            "compliance": compliance_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-pdf")
def generate_pdf(data: KYCResponse):
    try:
        html_content = f"""
        <html>
        <head><style>
        body {{ font-family: Arial, sans-serif; }}
        h1, h2 {{ color: #333; }}
        .risk {{ background: #f8d7da; padding: 10px; }}
        .fraud {{ background: #d4edda; padding: 10px; }}
        .compliance {{ background: #fff3cd; padding: 10px; }}
        </style></head>
        <body>
            <h1>KYC Analysis Report</h1>
            <h2>Risk Assessment</h2>
            <div class='risk'>
                <p>Risk Score: {data.risk['risk_score'] * 100}%</p>
                <p>Recommendation: {data.risk['recommendation']}</p>
            </div>

            <h2>Fraud Detection</h2>
            <div class='fraud'>
                <p>Fraud Detected: {"Yes" if data.fraud["fraud_detected"] else "No"}</p>
                <ul>
                    {''.join(f"<li>{reason}</li>" for reason in data.fraud["fraud_reasons"])}
                </ul>
            </div>

            <h2>Compliance Check</h2>
            <div class='compliance'>
                <ul>
                    {''.join(f"<li>{issue}</li>" for issue in data.compliance["compliance_issues"])}
                </ul>
            </div>
        </body>
        </html>
        """
        pdf_file = "kyc_report.pdf"
        pdfkit.from_string(html_content, pdf_file)
        return FileResponse(pdf_file, filename="KYC_Report.pdf", media_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
