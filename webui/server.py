"""FastAPI console. Run: ``python -m webui`` after ``source env.sh``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webui.llm import generate_reward, public_config, save_config
from webui.metrics import find_videos, preferred_series
from webui.preflight import check_recipe, sentry_alerts, vram_estimate
from webui.reward_check import apply_code, compose_file, diff_against_current, start_live_check, static_check
from webui.runner import (
    REPO_ROOT,
    RUNS_ROOT,
    delete_recipe,
    get_job,
    job_log,
    list_jobs,
    list_recipes,
    public_job,
    save_recipe,
    start_job,
    stop_job,
)
from webui.sandbox import FRAME_PATH, read_state, schema as sandbox_schema, write_command
from webui.schema import params_markdown, schema_payload

STATIC = Path(__file__).resolve().parent / "static"
PREVIEW_DIR = REPO_ROOT / "logs" / "webui" / "preview"

app = FastAPI(title="PΛN Dexterous Lab", version="0.1.0")
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


class RecipeIn(BaseModel):
    name: str | None = None
    task: str = "PAN-CoinHold-Apex-v0"
    play_task: str | None = None
    mode: str = "train"
    physics: str = "physx"
    object: str = "pan_coin_32mm"
    object2: str | None = None
    hand_pose: str = "palm_down_knuckle"
    cameras: str = "none"
    cli: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    chain: dict[str, Any] | None = None
    hydra: list[str] = Field(default_factory=list)


class RunIn(RecipeIn):
    mode: str = "train"


class LlmConfigIn(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_s: float | None = None


class GenerateIn(BaseModel):
    prompt: str
    task: str = "PAN-CoinHold-Apex-v0"
    retry_on_fail: bool = True


class ApplyIn(BaseModel):
    code: str
    entry: str | None = None
    live_check: bool = True
    task: str = "PAN-CoinHold-Apex-Play-v0"


class VramIn(BaseModel):
    num_envs: int = 512
    cameras: str = "none"
    width: int = 128
    height: int = 128


class SandboxCommandIn(BaseModel):
    targets_deg: dict[str, float] | None = None
    reset: bool | None = None
    pause: bool | None = None
    stop: bool | None = None
    reset_view: bool | None = None
    view: str | None = None
    camera: dict[str, Any] | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    page = STATIC / "index.html"
    if not page.is_file():
        return HTMLResponse("<p>missing webui/static/index.html</p>", status_code=500)
    return HTMLResponse(page.read_text())


@app.get("/api/schema")
def api_schema():
    return schema_payload()


@app.get("/api/docs/params")
def api_params_md():
    return {"markdown": params_markdown()}


@app.get("/api/recipes")
def api_recipes():
    return {"recipes": list_recipes()}


@app.post("/api/recipes")
def api_save_recipe(body: RecipeIn):
    name = body.name or body.task
    path = save_recipe(name, body.model_dump())
    return {"ok": True, "file": str(path)}


@app.delete("/api/recipes/{name}")
def api_delete_recipe(name: str):
    if not delete_recipe(name):
        raise HTTPException(404, "recipe not found")
    return {"ok": True}


@app.post("/api/preflight")
def api_preflight(body: RecipeIn):
    return {"alerts": check_recipe(body.model_dump())}


@app.post("/api/vram")
def api_vram(body: VramIn):
    n_cam = {"none": 0, "wrist_only": 1, "overhead_only": 1, "tri_view": 3}.get(body.cameras, 0)
    return vram_estimate(body.num_envs, n_cam, body.width, body.height)


@app.post("/api/runs")
def api_start(body: RunIn):
    recipe = body.model_dump()
    alerts = check_recipe(recipe)
    if any(a["level"] == "error" for a in alerts):
        raise HTTPException(status_code=400, detail={"alerts": alerts})
    try:
        job = start_job(recipe, mode=body.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job, "alerts": alerts}


@app.get("/api/runs")
def api_runs():
    return {"runs": list_jobs()}


@app.get("/api/runs/{job_id}")
def api_run(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "run not found")
    return public_job(job)


@app.delete("/api/runs/{job_id}")
def api_stop(job_id: str):
    try:
        return stop_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "run not found") from exc


@app.get("/api/runs/{job_id}/log")
def api_log(job_id: str, tail: int = 200):
    return {"text": job_log(job_id, tail=tail)}


@app.get("/api/runs/{job_id}/metrics")
def api_metrics(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "run not found")
    isaac = job.get("isaac_log_dir")
    series = preferred_series(isaac) if isaac else {}
    desired_kl = 0.01
    try:
        desired_kl = float(((job.get("recipe") or {}).get("overrides") or {}).get("agent.algorithm.desired_kl", 0.01))
    except (TypeError, ValueError):
        pass
    return {
        "series": series,
        "alerts": sentry_alerts(series, desired_kl=desired_kl),
        "videos": find_videos(isaac) if isaac else [],
        "isaac_log_dir": isaac,
    }


@app.get("/api/runs/{job_id}/video")
def api_video(job_id: str, rel: str):
    job = get_job(job_id)
    if not job or not job.get("isaac_log_dir"):
        raise HTTPException(404, "no isaac log dir")
    path = (Path(job["isaac_log_dir"]) / rel).resolve()
    root = Path(job["isaac_log_dir"]).resolve()
    if root not in path.parents and path != root:
        raise HTTPException(400, "bad path")
    if not path.is_file():
        raise HTTPException(404, "video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/exports")
def api_exports():
    root = REPO_ROOT / "logs" / "rsl_rl"
    items = []
    if root.is_dir():
        for onnx in sorted(root.rglob("exported/policy.onnx"), key=lambda p: p.stat().st_mtime, reverse=True):
            mapping = onnx.parent / "joint_map.json"
            items.append(
                {
                    "onnx": str(onnx.relative_to(REPO_ROOT)),
                    "map": str(mapping.relative_to(REPO_ROOT)) if mapping.is_file() else None,
                    "run": str(onnx.parent.parent.relative_to(REPO_ROOT)),
                    "mtime": onnx.stat().st_mtime,
                }
            )
    return {"exports": items[:20]}


@app.get("/api/sandbox/meta")
def api_sandbox_meta():
    return sandbox_schema()


@app.get("/api/sandbox/state")
def api_sandbox_state():
    return read_state()


@app.post("/api/sandbox/command")
def api_sandbox_command(body: SandboxCommandIn):
    return write_command(body.model_dump(exclude_none=True))


@app.get("/api/sandbox/frame")
def api_sandbox_frame():
    if not FRAME_PATH.is_file():
        raise HTTPException(404, "仿真还没出画面，Kit 启动大约要 30–90 秒。")
    return FileResponse(
        FRAME_PATH,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/preview/image")
def api_preview_image():
    png = PREVIEW_DIR / "scene.png"
    if not png.is_file():
        raise HTTPException(404, "还没有预览图，先点「场景预览」。")
    return FileResponse(png, media_type="image/png")


@app.get("/api/llm/config")
def api_llm_get():
    return public_config()


@app.put("/api/llm/config")
def api_llm_put(body: LlmConfigIn):
    return save_config(body.model_dump(exclude_none=True))


@app.post("/api/llm/generate")
def api_llm_generate(body: GenerateIn):
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt 为空")
    feedback = None
    last = None
    attempts = 3 if body.retry_on_fail else 1
    for _ in range(attempts):
        last = generate_reward(prompt, feedback=feedback)
        chk = static_check(last["code"])
        last["static"] = chk
        if chk["ok"]:
            proposed = compose_file(last["code"])
            last["proposed"] = proposed
            last["diff"] = diff_against_current(proposed)
            return last
        feedback = "\n".join(chk["errors"])
    last["proposed"] = compose_file(last["code"]) if last else ""
    last["diff"] = diff_against_current(last["proposed"]) if last else ""
    return last


@app.post("/api/llm/apply")
def api_llm_apply(body: ApplyIn):
    chk = static_check(body.code)
    if not chk["ok"]:
        raise HTTPException(400, {"static": chk})
    text = compose_file(body.code, entry=body.entry)
    apply_code(text)
    live = start_live_check(body.task) if body.live_check else None
    return {"ok": True, "static": chk, "live_job": live}


def main():
    import uvicorn

    uvicorn.run("webui.server:app", host="127.0.0.1", port=8090, reload=False)


if __name__ == "__main__":
    main()
