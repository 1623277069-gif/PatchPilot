FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir pytest==8.4.2
COPY scripts/sandbox_entry.py scripts/demo_check.py /opt/
ENTRYPOINT ["/usr/local/bin/python", "-I", "/opt/sandbox_entry.py"]
