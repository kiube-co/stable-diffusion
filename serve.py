import os.path
from os import getenv

import pickledb
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pusher import Pusher
from tinydb import TinyDB

from generators.text import TextGenerator
from repositories.query import Query

load_dotenv()

origins = [
    "http://localhost:3000",
]

db = TinyDB('db.json')
settings = pickledb.load('settings.db', False)

queries = db.table("queries")

app = FastAPI()
text_generator = TextGenerator()
pusher_client = Pusher(
    app_id=getenv("PUSHER_APP_ID"),
    key=getenv("PUSHER_KEY"),
    secret=getenv("PUSHER_SECRET"),
    cluster=getenv("PUSHER_CLUSTER")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"))

settings.set('busy', False)


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


def on_complete(query):
    try:
        pusher_client.trigger('tesseract-development', 'query-complete', {
            "content": query.as_json()
        })

    except:
        pass
    queries.insert(query.as_json())
    settings.set('busy', False)


@app.post("/queries", status_code=201)
async def add_query(query: Query, background_tasks: BackgroundTasks, response: Response):
    if os.path.exists(os.path.join("./images", query.filename())):
        response.status_code = 303
        return {
            "message": "Image already generated",
            "content": query.as_json()
        }
    else:
        if settings.get('busy') is True:
            response.status_code = 503
            return {
                "message": "Query running"
            }
        else:
            try:
                background_tasks.add_task(
                    text_generator.generate,
                    query,
                    lambda step, total_steps: on_step(step, total_steps),
                    lambda: on_complete(query)
                )
                settings.set('busy', True)
                return {
                    "message": "Query created"
                }
            except:
                settings.set('busy', False)
                response.status_code = 500
                return {
                    "message": "Error"
                }
