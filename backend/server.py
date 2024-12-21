from typing import Union

from fastapi import FastAPI
from inference import inference
from pydantic import BaseModel

app = FastAPI()

class InferencePayload(BaseModel):
    compound_smile: str
    protein_sequence:str



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/inference")
async def get_inference(payload: InferencePayload):
    predicted_class, confidence_scores = inference()
    return {"predicted_class":predicted_class, "confidence_score":confidence_scores}
    
