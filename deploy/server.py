from typing import Union

from fastapi import FastAPI, Response, status
from inference import inference
from pydantic import BaseModel

app = FastAPI()

class InferencePayload(BaseModel):
    compound_smile: str
    protein_sequence:str



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/inference")
async def get_inference(payload: InferencePayload, response: Response):
    try:
        predicted_class, confidence_scores = inference(payload.compound_smile,payload.protein_sequence)
    except Exception as error:
        print(error)
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        return{"message":f"Error while inferring: {error}"}
    return {"predicted_class":predicted_class.tolist(), "confidence_score":confidence_scores.tolist()}





    


