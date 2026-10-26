# Image "ml" : entraînement, évaluation, notebook, surveillance de la dérive, réentraînement
FROM python:3.12-slim

RUN useradd --create-home --uid 10001 artelio
WORKDIR /app
# PyTorch CPU (pas de GPU nécessaire), puis le reste des dépendances
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Le modèle de plongements est téléchargé À LA CONSTRUCTION : les conteneurs de la plateforme
# n'ont pas accès à Internet (réseau interne), et la version utilisée est figée dans l'image.
ENV HF_HOME=/opt/hf SENTENCE_TRANSFORMERS_HOME=/opt/hf
COPY scripts/telecharger_plongements.py /tmp/
RUN python /tmp/telecharger_plongements.py && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
COPY data ./data
COPY notebooks ./notebooks
COPY monitoring ./monitoring
COPY retrain ./retrain
COPY api ./api
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=/app
USER artelio
CMD ["python", "-m", "artelio_ia.entrainement"]
