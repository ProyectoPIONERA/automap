import os


_REQUIRED = {
    "HF": ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"),
    "WANDB": ("WANDB_API_KEY",),
}

def _get_any(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(f"Falta variable de entorno: {', '.join(names)}")

def setup_hf() -> None:
    """
    HuggingFace no necesita login si hay token en entorno.
    Solo validamos presencia y activamos transfer acelerado si está disponible.
    """
    _get_any(*_REQUIRED["HF"])
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Opcional: respeta caches efímeras si ya están exportadas por el job.sh
    # os.environ.setdefault("HF_HOME", "/tmp/hf")  # mejor hacerlo en job.sh

def setup_wandb(project: str | None = None, **init_kwargs):
    """
    W&B usa WANDB_API_KEY del entorno. No llamamos a wandb.login() para no persistir nada.
    """
    _get_any(*_REQUIRED["WANDB"])
    import wandb
    # Si WANDB_DIR está definido por el job, creamos el directorio (no persiste credenciales).
    wd = os.environ.get("WANDB_DIR")
    if wd:
        os.makedirs(wd, exist_ok=True)
    return wandb.init(project=project, **init_kwargs)

def setup_auth(wandb_project: str | None = None, **wandb_init_kwargs):
    """
    Punto único: llama desde cada script antes de usar HF/W&B.
    """
    setup_hf()
    run = None
    if wandb_project is not None:
        run = setup_wandb(project=wandb_project, **wandb_init_kwargs)
    return run

