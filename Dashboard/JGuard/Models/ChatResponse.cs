using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JGuard.Models;

public class ChatResponse
{
    [JsonPropertyName("reply")]
    public string Reply { get; set; } = string.Empty;

    [JsonPropertyName("blocked")]
    public bool Blocked { get; set; }

    [JsonPropertyName("triggered_defenses")]
    public List<string> TriggeredDefenses { get; set; } = new();

    [JsonPropertyName("decision")]
    public string? Decision { get; set; }

    [JsonPropertyName("harm_label")]
    public string? HarmLabel { get; set; }

    [JsonPropertyName("timestamp")]
    public string Timestamp { get; set; } = string.Empty;

    /// <summary>
    /// The prompt actually sent to the model after PII redaction and de-obfuscation.
    /// Differs from the user's typed prompt only when one of those defenses rewrote it.
    /// </summary>
    [JsonPropertyName("clean_prompt")]
    public string? CleanPrompt { get; set; }
}
