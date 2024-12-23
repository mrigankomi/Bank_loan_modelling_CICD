FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY bank_loan.py bank_loan.py
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app