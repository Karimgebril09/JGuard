using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JGuard.Models;

public class Session
{
    [JsonPropertyName("session_id")]
    public string SessionId { get; set; } = string.Empty;
    
    [JsonPropertyName("config")]
    public SessionConfig Config { get; set; } = new();
    
    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;
    
    [JsonPropertyName("last_active")]
    public string LastActive { get; set; } = string.Empty;
}

public class SessionHistory
{
    [JsonPropertyName("session_id")]
    public string SessionId { get; set; } = string.Empty;
    
    [JsonPropertyName("history")]
    public List<ChatMessage> History { get; set; } = new();
    
    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;
    
    [JsonPropertyName("last_active")]
    public string LastActive { get; set; } = string.Empty;

    [JsonPropertyName("meta")]
    public Dictionary<string, object> Meta { get; set; } = new();
}
