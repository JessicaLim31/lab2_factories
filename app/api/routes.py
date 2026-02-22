from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
import json



router = APIRouter()
topics_path = "data/topic_keywords.json"
emails_path = "data/emails.json"

class EmailRequest(BaseModel):
    subject: str
    body: str

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: int

class TopicCreate(BaseModel):
    new_topic: str
    description: str
    
    
class StoreEmailRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None

    
@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()


@router.post("/topics")
async def add_topics(request: TopicCreate):
    """Add New Topics"""
    try:
        with open(topics_path, "r") as f:
            topics = json.load(f)
    
        new_topic =  request.new_topic.strip()
        description = request.description.strip()
    
        if new_topic in topics:
            raise HTTPException(status_code=409, detail="Topic already exists")
    
        topics[new_topic] = {"description": description}
    
        with open(topics_path, "w") as f:
            json.dump(topics, f, indent=2)
    
        inference_service = EmailTopicInferenceService()
        info = inference_service.get_pipeline_info()
    
        return {
            "status": "Success", 
            "new_topic": new_topic, 
            "new_description": description,
            "topics": info["available_topics"]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/emails")
async def store_emails(request: StoreEmailRequest):
    try: 
        with open(emails_path, "r") as f:
            emails = json.load(f)
            
        new_email = {
            "id" : len(emails) + 1,
            "subject": request.subject.strip(),
            "body": request.body,
            "ground_truth": request.ground_truth
                    if request.ground_truth else None
        }
        emails.append(new_email)
            
        with open(emails_path, "w") as f:
            json.dump(emails,f,indent = 2)
        
        return {
            "status": "Email stored successfully.",
            "id": new_email["id"],
            "ground_truth": new_email["ground_truth"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

