FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt setup.py ./
COPY evaluator/ ./evaluator/
RUN pip install --user --no-cache-dir .

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY evaluator/ ./evaluator/
ENV PATH=/root/.local/bin:$PATH
ENTRYPOINT ["mlops-eval"]
