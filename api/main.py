import logging
from fastapi import FastAPI

from api.app.router.user_router import router as user_router
from api.app.scripts.create_tables import create_tables
from api.app.router.prompt_router import router as prompt_router


#create tables
create_tables()

logging.info("Start fastapi server")
app = FastAPI()

app.include_router(user_router)
app.include_router(prompt_router)
