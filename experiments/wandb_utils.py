import os
from datetime import datetime


def setup_wandb(args, script_name: str, extra_config: dict = None):
    if not getattr(args, "use_wandb", False):
        return None
    try:
        import wandb
    except Exception:
        print("[wandb] wandb is not installed. Run `pip install wandb`.")
        return None

    run_name = getattr(args, "wandb_run_name", "")
    if not run_name:
        run_name = f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cfg = vars(args).copy()
    if extra_config:
        cfg.update(extra_config)

    init_kwargs = {
        "project": getattr(args, "wandb_project", "cdc2026"),
        "name": run_name,
        "config": cfg,
    }
    entity = getattr(args, "wandb_entity", "")
    group = getattr(args, "wandb_group", "")
    if entity:
        init_kwargs["entity"] = entity
    if group:
        init_kwargs["group"] = group
    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[wandb] init failed: {exc}")
        return None
    return run


def log_rows_table(run, table_name: str, rows):
    if run is None or not rows:
        return
    try:
        import wandb
    except Exception:
        return
    cols = sorted({k for r in rows for k in r.keys()})
    data = [[r.get(c, None) for c in cols] for r in rows]
    run.log({table_name: wandb.Table(data=data, columns=cols)})


def log_summary_dict(run, summary: dict):
    if run is None or not summary:
        return
    run.log(summary)


def save_artifacts(run, paths):
    if run is None:
        return
    try:
        import wandb
    except Exception:
        return
    for p in paths:
        if p and os.path.exists(p):
            try:
                wandb.save(p)
            except Exception:
                pass


def finish_wandb(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
