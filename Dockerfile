FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY api/ ./api

EXPOSE 8000

# Command to start the API, including waiting for the database
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "api.main:app"]