FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

WORKDIR /code

# build-essential is needed for some chromadb/sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" 

COPY . .

# Build the vectorstore from whatever PDFs are in data/ at image build time.
# Render's free-tier disk is ephemeral — it does NOT persist across deploys —
# so the index has to be rebuilt into every image, not built once and kept.
# If you add/change PDFs later, you rebuild the image. There is no
# "update the running container" path on this setup.
RUN python ingest.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]

