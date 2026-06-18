using System;
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

    [JsonIgnore]
    public string LastActiveDisplay
    {
        get
        {
            if (DateTimeOffset.TryParse(LastActive, out var dto))
            {
                var local = dto.ToLocalTime();
                var span = DateTimeOffset.Now - local;

                if (Math.Abs(span.TotalSeconds) < 60)
                    return "just now";
                if (span.TotalMinutes < 60)
                    return $"{(int)span.TotalMinutes} minutes ago";
                if (span.TotalHours < 24)
                    return $"{(int)span.TotalHours} hours ago";
                if (span.TotalDays < 7)
                    return $"{(int)span.TotalDays} days ago";

                return local.ToString("MMM d, yyyy h:mm tt");
            }

            return LastActive;
        }
    }

    [JsonIgnore]
    public string LastActiveFull
    {
        get
        {
            if (DateTimeOffset.TryParse(LastActive, out var dto))
                return dto.ToLocalTime().ToString("f");
            return LastActive;
        }
    }
}

public class SessionListResponse
{
    [JsonPropertyName("sessions")]
    public List<Session> Sessions { get; set; } = new();
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
