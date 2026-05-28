FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --no-dev

# Copy patch script early so we can run it right after install
COPY scripts/patch_morph_kgc.sh ./scripts/patch_morph_kgc.sh
RUN chmod +x scripts/patch_morph_kgc.sh

# Apply morph-kgc compatibility patches:
# 1. pandas>=2.0: value_counts()[0] -> value_counts().iloc[0]
# 2. numpy>=2.0: np.NaN -> np.nan
RUN uv run bash scripts/patch_morph_kgc.sh

# Copy the rest of the project
COPY . .

# Copy env template so users know what variables are needed.
# The actual .env is mounted at runtime via docker-compose or -v flag.
COPY .env.example .env.example

# Use ENTRYPOINT so extra CLI flags (e.g. --shacl --sparql) can be passed
# via `docker run` or the docker-compose `command:` key.
ENTRYPOINT ["uv", "run", "python", "main.py"]
