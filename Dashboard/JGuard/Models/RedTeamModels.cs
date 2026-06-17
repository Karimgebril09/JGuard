using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JGuard.Models;

public class RedTeamLaunchRequest
{
    [JsonPropertyName("strategy")]
    public string Strategy { get; set; } = "custom";

    [JsonPropertyName("tool_framework")]
    public string? ToolFramework { get; set; }

    [JsonPropertyName("custom_attack_type")]
    public string? CustomAttackType { get; set; }

    [JsonPropertyName("custom_harm_type")]
    public string? CustomHarmType { get; set; }

    [JsonPropertyName("num_samples")]
    public int NumSamples { get; set; } = 5;

    [JsonPropertyName("judge_model")]
    public string? JudgeModel { get; set; }

    [JsonPropertyName("attacker_model")]
    public string? AttackerModel { get; set; }

    [JsonPropertyName("target_model")]
    public string? TargetModel { get; set; }

    [JsonPropertyName("judge_type")]
    public string? JudgeType { get; set; }

    [JsonPropertyName("attacker_type")]
    public string? AttackerType { get; set; }

    [JsonPropertyName("target_type")]
    public string? TargetType { get; set; }

    [JsonPropertyName("judge_api_key")]
    public string? JudgeApiKey { get; set; }

    [JsonPropertyName("attacker_api_key")]
    public string? AttackerApiKey { get; set; }

    [JsonPropertyName("target_api_key")]
    public string? TargetApiKey { get; set; }

    [JsonPropertyName("judge_base_url")]
    public string? JudgeBaseUrl { get; set; }

    [JsonPropertyName("attacker_base_url")]
    public string? AttackerBaseUrl { get; set; }

    [JsonPropertyName("target_base_url")]
    public string? TargetBaseUrl { get; set; }

    [JsonPropertyName("obfuscation_protection")]
    public bool ObfuscationProtection { get; set; }

    [JsonPropertyName("roleplay_protection")]
    public bool RoleplayProtection { get; set; }

    [JsonPropertyName("multi_turn_protection")]
    public bool MultiTurnProtection { get; set; }

    [JsonPropertyName("pii_protection")]
    public bool PiiProtection { get; set; }

    [JsonPropertyName("pii_strategy")]
    public string PiiStrategy { get; set; } = "mask";
}

public class RedTeamLaunchResponse
{
    [JsonPropertyName("campaign_id")]
    public string CampaignId { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("timestamp")]
    public string Timestamp { get; set; } = string.Empty;
}

public class RedTeamStatusResponse
{
    [JsonPropertyName("campaign_id")]
    public string CampaignId { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("log_lines")]
    public List<string> LogLines { get; set; } = new();

    [JsonPropertyName("progress_pct")]
    public int ProgressPct { get; set; }
}
