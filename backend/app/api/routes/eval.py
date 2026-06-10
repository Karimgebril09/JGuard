from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from backend.app.core import eval_store
from backend.app.models.eval_models import (
    AttackTrendPoint,
    CompareRequest,
    CompareResponse,
    EvalRunRecord,
    EvalSummaryResponse,
    VulnerabilityBreakdownResponse,
)


router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/summary", response_model=EvalSummaryResponse)
def summary() -> EvalSummaryResponse:
    return EvalSummaryResponse(**eval_store.get_summary())


@router.get("/vulnerability-breakdown", response_model=VulnerabilityBreakdownResponse)
def vulnerability_breakdown() -> VulnerabilityBreakdownResponse:
    return VulnerabilityBreakdownResponse(**eval_store.get_vulnerability_breakdown())


@router.get("/attack-trends", response_model=list[AttackTrendPoint])
def attack_trends() -> list[AttackTrendPoint]:
    return [AttackTrendPoint(**row) for row in eval_store.get_attack_trends()]


@router.get("/runs", response_model=list[EvalRunRecord])
def runs() -> list[EvalRunRecord]:
    return [EvalRunRecord(**row) for row in eval_store.get_runs()]


@router.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest) -> CompareResponse:
    try:
        return CompareResponse(**eval_store.compare_runs(payload.baseline_run_id, payload.compare_run_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/export")
def export(format: str = Query("json", pattern="^(csv|json)$")) -> Response:
    if format == "csv":
        content = eval_store.export_runs_as_csv()
        media_type = "text/csv"
        filename = "jguard_eval_runs.csv"
    else:
        content = eval_store.export_runs_as_json()
        media_type = "application/json"
        filename = "jguard_eval_runs.json"

    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(content=content, media_type=media_type, headers=headers)