from pydantic import BaseModel


class EvalSummaryResponse(BaseModel):
    total_campaigns: int
    avg_jailbreak_success_rate: float
    defense_blocked_sweeps_pct: float


class AttackTrendPoint(BaseModel):
    run_id: str
    success_rate: float


class EvalRunRecord(BaseModel):
    run_id: str
    timestamp: str
    target_model: str
    strategy: str
    defenses_active: str
    success_rate: float
    duration: str


class CompareRequest(BaseModel):
    baseline_run_id: str
    compare_run_id: str


class FloatDelta(BaseModel):
    base: float
    compare: float
    delta: float


class DurationPair(BaseModel):
    base: str
    compare: str


class CompareResponse(BaseModel):
    jailbreak_success_rate: FloatDelta
    assessment_duration: DurationPair
