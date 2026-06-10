# JGuard FastAPI Backend

This folder contains a backend scaffold for a future WinUI dashboard.

## What is included

- A FastAPI application entrypoint.
- Router templates for `system`, `chat`, `redteam`, and `eval`.
- Stub core modules that return mock data until system logic is ready.
- Stable API contracts using dedicated Pydantic model files.

## Folder structure

```text
backend/
  app/
    api/
      routes/
        system.py
        chat.py
        redteam.py
        eval.py
    core/
      defense_engine.py
      redteam_runner.py
      eval_store.py
    models/
    main.py
  requirements.txt
```

## Run locally

From the repository root:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload
```

## Endpoint groups

- `GET /api/system/status`
- `POST /api/system/config`
- `POST /api/chat/foundational`
- `POST /api/chat/agent`
- `POST /api/chat/clear`
- `POST /api/redteam/launch`
- `GET /api/redteam/status/{campaign_id}`
- `POST /api/redteam/stop/{campaign_id}`
- `GET /api/eval/summary`
- `GET /api/eval/vulnerability-breakdown`
- `GET /api/eval/attack-trends`
- `GET /api/eval/runs`
- `POST /api/eval/compare`
- `GET /api/eval/export?format=csv|json`

## Utility endpoints

- `GET /api/health`
- `GET /api/health/ping`

## Sample request

Example for `POST /api/chat/foundational`:

```json
{
  "prompt": "test prompt",
  "local_llm": true,
  "llm_api_key": "",
  "llm_type": "llama3.1:8b",
  "obfuscation_protection": true,
  "multi_turn_protection": true,
  "roleplay_protection": true,
  "history": []
}
```

## Wiring real logic later

1. Replace stub behavior in `backend/app/core/defense_engine.py` with real runtime calls.
2. Replace campaign orchestration in `backend/app/core/redteam_runner.py`.
3. Replace mock eval storage in `backend/app/core/eval_store.py` with actual persistence.
4. Keep routers and models stable so the WinUI client does not need to change.
