from os import getenv

from dotenv import load_dotenv
from fastapi import FastAPI
from pusher import Pusher

from generators.text import TextGenerator
from repositories.query import Query

load_dotenv()

app = FastAPI()
text_generator = TextGenerator()
pusher_client = Pusher(
    app_id=getenv("PUSHER_APP_ID"),
    key=getenv("PUSHER_KEY"),
    secret=getenv("PUSHER_SECRET"),
    cluster=getenv("PUSHER_CLUSTER")
)


@app.get("/")
async def root():
    return {
        "name": "Stable Diffusion Service",
        "version": "0.0.1"
    }


def on_step(step, total_steps):
    try:
        pusher_client.trigger('tesseract-development', 'query-progress', {
            "step": step,
            "total_steps": total_steps
        })
    except:
        pass


@app.post("/queries")
async def add_query(query: Query):
    text_generator.generate(
        query,
        lambda step, total_steps: on_step(step, total_steps)
    )
    return {
        "message": "Query finished"
    }
