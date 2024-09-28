FROM python:3.11

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./app /app

WORKDIR /app

CMD ["python", "-c", "while True: pass"]
