from pydantic import BaseModel


class EvalSummaryResponse(BaseModel):
    total_campaigns: int
    avg_jailbreak_success_rate: float
    critical_issues_found: int
    defense_blocked_sweeps_pct: float


class VulnerabilityBreakdownResponse(BaseModel):
    critical: int
    high: int
    medium: int
    low: int


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
    vulnerabilities: int
    duration: str


class CompareRequest(BaseModel):
    baseline_run_id: str
    compare_run_id: str


class FloatDelta(BaseModel):
    base: float
    compare: float
    delta: float


class IntDelta(BaseModel):
    base: int
    compare: int
    delta: int


class DurationPair(BaseModel):
    base: str
    compare: str


class CompareResponse(BaseModel):
    jailbreak_success_rate: FloatDelta
    critical_vulnerabilities: IntDelta
    total_vulnerabilities: IntDelta
    assessment_duration: DurationPair
