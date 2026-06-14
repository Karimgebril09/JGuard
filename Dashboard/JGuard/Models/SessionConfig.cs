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
    
    [JsonPropertyName("obfuscation_protection")]
    public bool ObfuscationProtection { get; set; }
    
    [JsonPropertyName("multi_turn_protection")]
    public bool MultiTurnProtection { get; set; }
    
    [JsonPropertyName("roleplay_protection")]
    public bool RoleplayProtection { get; set; }
    
    [JsonPropertyName("pii_protection")]
    public bool PiiProtection { get; set; }
    
    [JsonPropertyName("pii_strategy")]
    public string PiiStrategy { get; set; } = "mask"; // "mask", "encrypt", or "block"

    // Helper properties for UI binding
    [JsonIgnore]
    public string SourceText => LocalLlm ? "Local (Ollama)" : "Cloud API";
    
    [JsonIgnore]
    public string ModeText => ChatMode.ToUpper();
    
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
