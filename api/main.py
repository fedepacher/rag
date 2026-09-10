import logging
from fastapi import FastAPI

from api.app.router.user_router import router as user_router
from api.app.router.prompt_router import router as prompt_router


# The schema is created by a one-shot step in the container entrypoint, before gunicorn
# forks its workers. Doing it here ran the DDL once per worker in parallel and raced.
logging.info("Start fastapi server")
app = FastAPI()

app.include_router(user_router)
app.include_router(prompt_router)
