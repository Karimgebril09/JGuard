using System.Text.Json.Serialization;

namespace JGuard.Models;

public class DeleteSessionResponse
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    
    [JsonPropertyName("session_id")]
    public string SessionId { get; set; } = string.Empty;
}
