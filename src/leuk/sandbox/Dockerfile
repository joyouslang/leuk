FROM python:3.11-slim

# Install minimal runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user that the container runs as by default.
# The UID/GID are overridden at runtime via --user=<host-uid>:<host-gid>
# so that files created inside the container are owned by the calling user.
RUN groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid sandbox --shell /bin/sh --create-home sandbox

USER sandbox
WORKDIR /workspace
