FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Προστέθηκε το scikit-learn
RUN pip install pandas lightgbm entsoe-py scikit-learn

COPY main.py /app/main.py

CMD ["python", "main.py"]