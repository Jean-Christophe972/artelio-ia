# Image de l'API de qualification (déployée sur Railway et sur la plateforme locale)
FROM python:3.12-slim

RUN useradd --create-home --uid 10001 artelio
WORKDIR /app
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements-api.txt

# Modèle de plongements figé dans l'image (aucun téléchargement au démarrage, réseau interne)
ENV HF_HOME=/opt/hf SENTENCE_TRANSFORMERS_HOME=/opt/hf
COPY scripts/telecharger_plongements.py /tmp/
RUN python /tmp/telecharger_plongements.py && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY api ./api
# Champion exporté : utilisé quand MODELE_SOURCE=local (ex. Railway, sans accès au registre)
COPY models/champion ./models/champion

ENV PYTHONUNBUFFERED=1 MODELE_SOURCE=local MODELE_DOSSIER=/app/models/champion PORT=8000
USER artelio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/pret', timeout=3)"
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-server-header"]
