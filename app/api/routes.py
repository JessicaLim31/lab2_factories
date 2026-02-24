import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Literal
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
import json


router = APIRouter()
topics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "topic_keywords.json")
emails_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "emails.json")

class EmailRequest(BaseModel):
    subject: str
    body: str
    mode: Optional[str] = None

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Optional[Dict[str, float]] = None
    features: Dict[str, Any]
    available_topics: Optional[List[str]] = None

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

    
''' @router.post("/emails/classify", response_model=EmailClassificationResponse)
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
'''

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


## Assignment
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
        
        if request.ground_truth:
            with open(topics_path, "r") as f:
                topics = json.load(f)
            if request.ground_truth not in topics:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ground_truth. Valid topics: {list(topics.keys())}"
                )
                
        inference_service = EmailTopicInferenceService()
        email_obj = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email_obj)
        
        new_email = {
            "id" : len(emails) + 1,
            "subject": request.subject.strip(),
            "body": request.body,
            "ground_truth": request.ground_truth
                    if request.ground_truth else None,
            "embedding": result["features"].get("email_embeddings_average_embedding")
        }
        emails.append(new_email)
            
        with open(emails_path, "w") as f:
            json.dump(emails,f,indent = 2)
        
        return {
            "status": "Email stored successfully.",
            "id": new_email["id"],
            "ground_truth": new_email["ground_truth"],
            "embedding": new_email["embedding"] is not None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        
        if request.mode == "email":
            with open (emails_path, "r") as f:
                stored_emails = json.load(f)
            label = [g for g in stored_emails if g.get("ground_truth")]    
            
            if not label:
                raise HTTPException(status_code=400, detail="No emails with ground truth stored yet.")
            
            result = inference_service.classify_email(email)    
            best_label, best_score = inference_service.model.predict_emails(result["features"], label)
            
            if best_label is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No similar email found."
                )
            
            return EmailClassificationResponse(
                predicted_topic=best_label,
                topic_scores={best_label: best_score},
                features=result["features"],
                available_topics=list(set(e["ground_truth"] for e in label)) 
            )
        
        result = inference_service.classify_email(email)
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

## Assignment

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

