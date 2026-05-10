FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install kafka-python psycopg2-binary torch numpy

CMD ["python", "k_d.py"]