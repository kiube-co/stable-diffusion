from fastapi import FastAPI

from generators.text import TextGenerator
from repositories.query import Query

app = FastAPI()
text_generator = TextGenerator()


@app.get("/")
async def root():
    return {
        "name": "Stable Diffusion Service",
        "version": "0.0.1"
    }


@app.post("/queries")
async def add_query(query: Query):
    text_generator.generate(query)
    return {
        "message": "Query finished"
    }
