from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from typing import Dict, List, Optional, Any
import os
from pydantic import EmailStr, Extra
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load KYC rules
def load_rules():
    with open("kyc_rules.json", "r") as file:
        return json.load(file)

RULES = load_rules()

# Azure OpenAI Model Configuration
llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    temperature=0.2,
)

# Models
class KYCData(BaseModel):
    full_name: Optional[str] = None
    dob: Optional[str] = None
    nationality: Optional[str] = None
    id_number: Optional[str] = None
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    transaction_history: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        extra = Extra.allow  # Allow extra fields in the JSONs

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
async def call_azure_openai(prompt: str):
    response = llm.invoke(prompt)

    # Extract JSON from the response text using regex
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON extracted", "raw_response": response.content}
    
    return {"error": "No JSON found", "raw_response": response.content}

async def check_risk(kyc_json: dict):
    prompt_template = PromptTemplate(
        input_variables=["kyc_data", "rules"],
        template=(
            "You are an expert in risk assessment. Analyze the risk for the following KYC data:\n\n"
            "{kyc_data}\n\n"
            "Here are some predefined risk assessment rules:\n{rules}\n\n"
            "Return your response in JSON format:\n"
            "{\"risk_score\": 0.65, \"recommendation\": \"Your recommendation here\"}"
        ),
    )
    
    formatted_prompt = prompt_template.format(
        kyc_data=json.dumps(kyc_json, indent=2),
        rules=json.dumps(RULES["risk_analysis"], indent=2)
    )
    print("Formatted Prompt:", formatted_prompt)  # Debugging print
    response = await call_azure_openai(formatted_prompt)

    # Ensure required keys exist
    risk_score = response.get("risk_score", 0.0)
    recommendation = response.get("recommendation", "No recommendation provided")

    return {"risk_score": risk_score, "recommendation": recommendation}

async def check_fraud(kyc_json: dict):
    prompt_template = PromptTemplate(
        input_variables=["kyc_data", "rules"],
        template=(
            "You are an expert in fraud detection. Analyze the following KYC data for fraud risk:\n\n"
            "{kyc_data}\n\n"
            "Here are some predefined fraud detection rules:\n{rules}\n\n"
            "Return your response in JSON format:\n"
            "{\"fraud_detected\": false, \"fraud_reasons\": []}"
        ),
    )
    
    formatted_prompt = prompt_template.format(
        kyc_data=json.dumps(kyc_json, indent=2),
        rules=json.dumps(RULES["fraud_detection"], indent=2)
    )
    
    response = await call_azure_openai(formatted_prompt)
    return response

async def check_compliance(kyc_json: dict):
    prompt_template = PromptTemplate(
        input_variables=["kyc_data", "rules"],
        template=(
            "You are a compliance officer. Check the following KYC data for regulatory compliance:\n\n"
            "{kyc_data}\n\n"
            "Here are some predefined compliance rules:\n{rules}\n\n"
            "Return your response in JSON format:\n"
            "{\"compliance_flags\": [], \"compliance_issues\": []}"
        ),
    )
    
    formatted_prompt = prompt_template.format(
        kyc_data=json.dumps(kyc_json, indent=2),
        rules=json.dumps(RULES["compliance_rules"], indent=2)
    )
    
    response = await call_azure_openai(formatted_prompt)
    return response

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "KYC Analysis API is running"}

@app.post("/analyze", response_model=KYCResponse)
async def analyze_kyc(request: KYCRequest):
    try:
        kyc_json = request.data
        print("Received KYC Data:", kyc_json)  # Debugging print

        risk_result, fraud_result, compliance_result = await asyncio.gather(
            check_risk(kyc_json),
            check_fraud(kyc_json),
            check_compliance(kyc_json)
        )

        print("Risk Result:", risk_result)  # Debugging print
        print("Fraud Result:", fraud_result)  # Debugging print
        print("Compliance Result:", compliance_result)  # Debugging print

        return {
            "risk": risk_result,
            "fraud": fraud_result,
            "compliance": compliance_result
        }
    except Exception as e:
        print("Error:", str(e))  # Debugging print
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
