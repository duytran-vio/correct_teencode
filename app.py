import uvicorn
from fastapi import FastAPI
from correct_teencode import correct_teencode
from correct_telex import TelexErrorCorrector
from pydantic import BaseModel

telexCorrector = TelexErrorCorrector()

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
