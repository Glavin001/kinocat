# Container for the JAX trainer. Built once locally; the host's Node
# driver shells out via `docker run` and mounts the per-round tmp dir
# so trial npz + params JSON flow through host filesystem without
# leaking anything into the host's Python.
#
# Build:  docker build -t kinocat-jax-trainer:latest demos/scripts/python
#         (or: pnpm run train:docker-build)
#
# Run (handled automatically by `pnpm run train -- --trainer=python --docker`):
#   docker run --rm \
#     -v /tmp/kinocat-jax-XYZ:/data \
#     -v $(pwd)/demos/scripts/python:/work \
#     -w /work \
#     --user "$(id -u):$(id -g)" \
#     kinocat-jax-trainer:latest \
#     python -m train_fit --trials /data/round-0/trials.npz ...

FROM python:3.11-slim

# JAX wheels need libgomp; jaxlib also benefits from a recent libstdc++
# already present in the slim image. No other system deps required for
# CPU-only inference.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install Python deps. We copy ONLY requirements.txt at this layer so
# changes to *.py don't bust the (slow) pip install layer. The source
# files are mounted at runtime — no need to bake them in.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# Silence noisy JAX banner on every spawn.
ENV JAX_PLATFORMS=cpu \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Default to a help message if no command is given.
CMD ["python", "-c", "print('kinocat-jax-trainer image OK. Invoke via `python -m train_fit ...`')"]
