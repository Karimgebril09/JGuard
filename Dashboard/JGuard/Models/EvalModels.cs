using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JGuard.Models;

/// <summary>GET /api/eval/summary — the top stat cards.</summary>
public class EvalSummaryResponse
{
    [JsonPropertyName("total_campaigns")]
    public int TotalCampaigns { get; set; }

    [JsonPropertyName("avg_jailbreak_success_rate")]
    public double AvgJailbreakSuccessRate { get; set; }

    [JsonPropertyName("defense_blocked_sweeps_pct")]
    public double DefenseBlockedSweepsPct { get; set; }
}

/// <summary>GET /api/eval/attack-trends — one point per historical run (line chart).</summary>
public class EvalAttackTrend
{
    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = string.Empty;

    [JsonPropertyName("success_rate")]
    public double SuccessRate { get; set; }
}

/// <summary>GET /api/eval/runs — a row in the run history log table.</summary>
public class EvalRun
{
    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = string.Empty;

    [JsonPropertyName("timestamp")]
    public string Timestamp { get; set; } = string.Empty;

    [JsonPropertyName("target_model")]
    public string TargetModel { get; set; } = string.Empty;

    [JsonPropertyName("strategy")]
    public string Strategy { get; set; } = string.Empty;

    [JsonPropertyName("defenses_active")]
    public string DefensesActive { get; set; } = string.Empty;

    [JsonPropertyName("success_rate")]
    public double SuccessRate { get; set; }

    [JsonPropertyName("duration")]
    public string Duration { get; set; } = string.Empty;

    // Optional vulnerability breakdown. Defaults to 0 when the backend omits these,
    // so the severity pie chart, the "Critical Issues" card, and the per-run
    // vulnerabilities column are populated straight from the server when available.
    [JsonPropertyName("critical_vulnerabilities")]
    public int CriticalVulnerabilities { get; set; }

    [JsonPropertyName("high_vulnerabilities")]
    public int HighVulnerabilities { get; set; }

    [JsonPropertyName("medium_vulnerabilities")]
    public int MediumVulnerabilities { get; set; }

    [JsonPropertyName("low_vulnerabilities")]
    public int LowVulnerabilities { get; set; }

    [JsonIgnore]
    public int TotalVulnerabilities =>
        CriticalVulnerabilities + HighVulnerabilities + MediumVulnerabilities + LowVulnerabilities;
}

/// <summary>POST /api/eval/compare body.</summary>
public class EvalCompareRequest
{
    [JsonPropertyName("baseline_run_id")]
    public string BaselineRunId { get; set; } = string.Empty;

    [JsonPropertyName("compare_run_id")]
    public string CompareRunId { get; set; } = string.Empty;
}

/// <summary>POST /api/eval/compare response — the "Analyze Delta" result.</summary>
public class EvalCompareResponse
{
    [JsonPropertyName("jailbreak_success_rate")]
    public EvalCompareFloatMetric JailbreakSuccessRate { get; set; } = new();

    [JsonPropertyName("critical_vulnerabilities")]
    public EvalCompareIntMetric CriticalVulnerabilities { get; set; } = new();

    [JsonPropertyName("total_vulnerabilities")]
    public EvalCompareIntMetric TotalVulnerabilities { get; set; } = new();

    [JsonPropertyName("assessment_duration")]
    public EvalCompareDurationMetric AssessmentDuration { get; set; } = new();
}

public class EvalCompareFloatMetric
{
    [JsonPropertyName("base")]
    public double Base { get; set; }

    [JsonPropertyName("compare")]
    public double Compare { get; set; }

    [JsonPropertyName("delta")]
    public double Delta { get; set; }
}

public class EvalCompareIntMetric
{
    [JsonPropertyName("base")]
    public int Base { get; set; }

    [JsonPropertyName("compare")]
    public int Compare { get; set; }

    [JsonPropertyName("delta")]
    public int Delta { get; set; }
}

public class EvalCompareDurationMetric
{
    [JsonPropertyName("base")]
    public string Base { get; set; } = string.Empty;

    [JsonPropertyName("compare")]
    public string Compare { get; set; } = string.Empty;
}
