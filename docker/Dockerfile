# Build from the repository root:
#   docker build -f docker/Dockerfile -t rsi-audit-worker:local .
# For a portable frozen build, supply a verified base-image digest via
# --build-arg PYTHON_BASE=python:3.12-slim@sha256:... . Execution always resolves
# this resulting image to its immutable image ID and records that identity.
ARG PYTHON_BASE=python:3.12-slim
FROM ${PYTHON_BASE}
LABEL org.rsi-audit.worker.protocol="1"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 HOME=/tmp
COPY --chmod=0444 rsi_audit/wire.py /opt/rsi_audit/wire.py
COPY --chmod=0444 docker/worker.py /opt/rsi_audit/worker.py
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python3", "-I", "/opt/rsi_audit/worker.py"]
