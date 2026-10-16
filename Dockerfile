FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY app.py .

# Entraîne le modèle à la construction de l'image (les CSV sont dans data/)
RUN python -m winequality.train --data-dir data --output models/wine_svm.joblib

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
