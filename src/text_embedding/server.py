from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]

@app.post("/custom/embeddings")
def create_embeddings(req: EmbeddingRequest):
    vectors = model.encode(req.input)
    data = [
            {"embedding": vec.tolist(), "index": i, "object": "embedding"}
            for i, vec in enumerate(vectors)
        ]
    
    return {
        "data": data,
        "model": req.model,
        "object": type(data)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)