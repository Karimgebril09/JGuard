using System;
using Microsoft.UI.Xaml;
using System.Text.Json.Serialization;

namespace JGuard.Models;

public class SessionConfig
{
    [JsonPropertyName("chat_mode")]
    public string ChatMode { get; set; } = "foundational"; // "foundational" or "agent"
    
    [JsonPropertyName("local_llm")]
    public bool LocalLlm { get; set; } = true; // true for Ollama, false for closed-source
    
    [JsonPropertyName("llm_api_key")]
    public string LlmApiKey { get; set; } = string.Empty;
    
    [JsonPropertyName("llm_type")]
    public string LlmType { get; set; } = string.Empty; // model name or provider model id

    [JsonPropertyName("llm_base_url")]
    public string LlmBaseUrl { get; set; } = string.Empty;

    [JsonPropertyName("obfuscation_protection")]
    public bool ObfuscationProtection { get; set; }
    
    [JsonPropertyName("multi_turn_protection")]
    public bool MultiTurnProtection { get; set; }
    
    [JsonPropertyName("roleplay_protection")]
    public bool RoleplayProtection { get; set; }
    
    [JsonPropertyName("pii_protection")]
    public bool PiiProtection { get; set; }
    
    [JsonPropertyName("pii_strategy")]
    public string PiiStrategy { get; set; } = "mask"; // "mask", "hash", "block", "partial masking"

    // Helper properties for UI binding
    [JsonIgnore]
    public bool IsAgent => string.Equals(ChatMode, "agent", StringComparison.OrdinalIgnoreCase);

    // Agent sessions have no user-supplied LLM source/model (the backend returns null),
    // so surface a friendly label instead of a blank field.
    [JsonIgnore]
    public string ModelDisplay => IsAgent
        ? "Agent System"
        : (string.IsNullOrWhiteSpace(LlmType) ? "Unknown Model" : LlmType);

    [JsonIgnore]
    public string SourceText => IsAgent ? "Agent Tooling" : (LocalLlm ? "Local (Ollama)" : "Cloud API");

    [JsonIgnore]
    public string ModeText => (ChatMode ?? string.Empty).ToUpper();
    
    [JsonIgnore]
    public Visibility ObfuscationVisibility => ObfuscationProtection ? Visibility.Visible : Visibility.Collapsed;
    
    [JsonIgnore]
    public Visibility MultiTurnVisibility => MultiTurnProtection ? Visibility.Visible : Visibility.Collapsed;
    
    [JsonIgnore]
    public Visibility RoleplayVisibility => RoleplayProtection ? Visibility.Visible : Visibility.Collapsed;
    
    [JsonIgnore]
    public Visibility PiiVisibility => PiiProtection ? Visibility.Visible : Visibility.Collapsed;
    
    [JsonIgnore]
    public Visibility NoDefensesVisibility => (!ObfuscationProtection && !MultiTurnProtection && !RoleplayProtection && !PiiProtection) ? Visibility.Visible : Visibility.Collapsed;
}
