using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JGuard.Models;

public class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = string.Empty;

    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;
}

public class ChatRequest
{
    [JsonPropertyName("prompt")]
    public string Prompt { get; set; } = string.Empty;
    
    [JsonPropertyName("local_llm")]
    public bool LocalLlm { get; set; }
    
    [JsonPropertyName("llm_api_key")]
    public string LlmApiKey { get; set; } = string.Empty;
    
    [JsonPropertyName("llm_type")]
    public string LlmType { get; set; } = string.Empty;
    
    [JsonPropertyName("obfuscation_protection")]
    public bool ObfuscationProtection { get; set; }
    
    [JsonPropertyName("multi_turn_protection")]
    public bool MultiTurnProtection { get; set; }
    
    [JsonPropertyName("roleplay_protection")]
    public bool RoleplayProtection { get; set; }
    
    [JsonPropertyName("pii_protection")]
    public bool PiiProtection { get; set; }
    
    [JsonPropertyName("history")]
    public List<ChatMessage> History { get; set; } = new();
}
