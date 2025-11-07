from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s:%(levelno)d %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
models = {
    "intfloat/multilingual-e5-base": SentenceTransformer("intfloat/multilingual-e5-base"),
}

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, list[str]]

@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    if req.model not in models.keys():
        return HTTPException(status_code=400, detail="Model not supported.")
    logger.info(f"Creating embeddings using model: {req.model}")
    vectors = models[req.model].encode(req.input, normalize_embeddings=True)
    data = [
            {"embedding": vec.tolist(), "index": i, "object": "embedding"}
            for i, vec in enumerate(vectors)
        ]
    logger.info("Embeddings created successfully.")
    return {
        "data": data,
        "model": req.model,
        "object": "list"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.text_embedding.server:app", host="0.0.0.0", port=8001, reload_dirs=["src/text_embedding"])