import uvicorn
from fastapi import FastAPI
from correct_teencode import correct_teencode
from correct_telex import TelexErrorCorrector
from pydantic import BaseModel
from accent import get_accented
from correct_close_character import correct_close_character_sent
import requests
import os



app = FastAPI()


class Request(BaseModel):
    text: str


@app.post("/correct-teencode")
def teencode(data: Request):
    data = data.dict()
    corrected = correct_teencode(data["text"])
    return {"result": corrected}


@app.post("/correct-telex")
def telex(data: Request):
    data = data.dict()
    corrected = telexCorrector.fix_telex_sentence(data["text"])
    return {"result": corrected}


@app.post("/correct-close")
def close(data: Request):
    data = data.dict()
    corrected = correct_close_character_sent(data["text"])
    return {"result": corrected}


@app.post("/correct-accent")
def accent(data: Request):
    data = data.dict()
    response = requests.post(os.environ["ACCENT_URL"], json={"text": data["text"]}).json()
    corrected = response["result"]
    return {"result": corrected}


if __name__ == "__main__":
    telexCorrector = TelexErrorCorrector()
    uvicorn.run(app, host="0.0.0.0", port=5000)
