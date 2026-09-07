FROM python:3.11-slim

ENV PIP_CACHE_DIR=/root/.cache/pip

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models logs reports/figures

EXPOSE 8501

CMD ["sh", "-c", "python main.py && streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]