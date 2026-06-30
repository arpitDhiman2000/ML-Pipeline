# DagsHub setup — DVC remote + hosted MLflow

DagsHub gives this project a **free cloud DVC remote** (so data versions are
pushable/pullable and browsable) and a **free hosted MLflow tracking server**
(so every training run from Sprint 2 onward shows up in a web dashboard with no
server to run yourself).

You do the account steps (your credentials); the code is already wired.

---

## 1. Create / connect the DagsHub repo (one time)

1. Sign in at <https://dagshub.com>.
2. **Create → Connect a repository → GitHub**, and pick
   `arpitDhiman2000/ML-Pipeline` (or **Create** a new DagsHub repo of the same
   name). Note your **DagsHub username** and **repo name** — they may differ
   from GitHub.
3. Get a token: **Your avatar → Settings → Tokens**, or the green **Remote**
   button on the repo page shows ready-made credentials.

Your three endpoints (from the repo's **Remote** button):
* MLflow: `https://dagshub.com/<user>/<repo>.mlflow`
* DVC:    `https://dagshub.com/<user>/<repo>.dvc`

---

## 2. Configure MLflow (via `.env`)

```powershell
# from the repo root
Copy-Item .env.example .env
```
Edit `.env` and set (using your real values):
```
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```
`.env` is gitignored — the token never leaves your machine.

Verify it works (logs a tiny run you can see in the **Experiments** tab):
```powershell
uv run td mlflow-smoke
```

---

## 3. Configure the DVC remote (token stays local)

```powershell
# point DVC at DagsHub and make it the default remote
uv run dvc remote add dagshub https://dagshub.com/<user>/<repo>.dvc
uv run dvc remote default dagshub

# credentials -> .dvc/config.local (gitignored), NOT committed
uv run dvc remote modify dagshub --local auth basic
uv run dvc remote modify dagshub --local user <user>
uv run dvc remote modify dagshub --local password <your-dagshub-token>

# push the versioned data/artifacts to the cloud
uv run dvc push
```

After `dvc push`, open the repo's **Data** tab on DagsHub to browse the
versioned `flows.parquet`, processed splits, and artifacts.

> The `dvc remote add`/`default` lines write to `.dvc/config` (safe to commit —
> no secrets). The `--local` lines write to `.dvc/config.local` (gitignored).

---

## What ends up where

| Surface | Lives on | How |
|---------|----------|-----|
| Code    | GitHub + DagsHub mirror | `git push` |
| Data / model artifacts | DagsHub DVC remote | `dvc push` |
| Experiments (params/metrics/artifacts) | DagsHub MLflow | automatic from Sprint 2 training |
| Deployment (ECR/EC2) | AWS | Sprint 5 |
