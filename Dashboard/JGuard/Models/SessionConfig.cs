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
    public string PiiStrategy { get; set; } = "mask"; // "mask", "hash", "block", "partial"

    // Agent-only extra defenses (sent as false for foundational sessions)
    [JsonPropertyName("web_search_protection")]
    public bool WebSearchProtection { get; set; }

    [JsonPropertyName("code_execution_protection")]
    public bool CodeExecutionProtection { get; set; }

    [JsonPropertyName("rag_protection")]
    public bool RagProtection { get; set; }

    [JsonPropertyName("email_protection")]
    public bool EmailProtection { get; set; }

    [JsonPropertyName("document_protection")]
    public bool DocumentProtection { get; set; }

    [JsonPropertyName("code_deep_check")]
    public bool CodeDeepCheck { get; set; }

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
    public Visibility WebSearchVisibility => WebSearchProtection ? Visibility.Visible : Visibility.Collapsed;

    [JsonIgnore]
    public Visibility CodeExecutionVisibility => CodeExecutionProtection ? Visibility.Visible : Visibility.Collapsed;

    [JsonIgnore]
    public Visibility RagVisibility => RagProtection ? Visibility.Visible : Visibility.Collapsed;

    [JsonIgnore]
    public Visibility EmailVisibility => EmailProtection ? Visibility.Visible : Visibility.Collapsed;

    [JsonIgnore]
    public Visibility DocumentVisibility => DocumentProtection ? Visibility.Visible : Visibility.Collapsed;

    [JsonIgnore]
    public Visibility CodeDeepCheckVisibility => CodeDeepCheck ? Visibility.Visible : Visibility.Collapsed;

    [JsonIgnore]
    public Visibility NoDefensesVisibility => (!ObfuscationProtection && !MultiTurnProtection && !RoleplayProtection && !PiiProtection
        && !WebSearchProtection && !CodeExecutionProtection && !RagProtection && !EmailProtection && !DocumentProtection && !CodeDeepCheck) ? Visibility.Visible : Visibility.Collapsed;
}
