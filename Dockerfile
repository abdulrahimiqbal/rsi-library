FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RSI_DB=/data/gateway.db

RUN groupadd --gid 10001 rsi \
    && useradd --uid 10001 --gid 10001 --no-create-home rsi \
    && mkdir /app /data \
    && chown 10001:10001 /data

WORKDIR /app
COPY rsi_library/ /app/rsi_library/
USER 10001:10001
ENTRYPOINT ["python", "-m", "rsi_library.gateway"]
CMD ["hosted"]
