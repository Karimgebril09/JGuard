using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using JGuard.Models;

namespace JGuard.Services;

public class JGuardApiService
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private string? _activeSessionId;

    public JGuardApiService(string baseUrl)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _httpClient = new HttpClient();
    }

    /// <summary>
    /// Get the currently active session ID
    /// </summary>
    public string? GetActiveSessionId => _activeSessionId;

    /// <summary>
    /// Set the active session ID
    /// </summary>
    public void SetActiveSessionId(string sessionId)
    {
        _activeSessionId = sessionId;
    }

    /// <summary>
    /// Creates a new session with the specified configuration
    /// </summary>
    public async Task<Session?> CreateSessionAsync(SessionConfig config)
    {
        try
        {
            var requestBody = new { config };
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/sessions", requestBody);
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<Session>();
                if (result != null)
                {
                    _activeSessionId = result.SessionId;
                }
                return result;
            }
            else
            {
                System.Diagnostics.Debug.WriteLine($"API Error creating session: {response.StatusCode}");
                return null;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error creating session: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Sends a chat message to the active session
    /// </summary>
    public async Task<ChatResponse?> SendChatAsync(string sessionId, string prompt)
    {
        try
        {
            var requestBody = new { prompt };
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/sessions/{sessionId}/chat", requestBody);
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<ChatResponse>();
                return result;
            }
            else
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                System.Diagnostics.Debug.WriteLine($"API Error: {response.StatusCode}. {errorContent}");
                return null;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error sending chat: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Gets the conversation history for a session
    /// </summary>
    public async Task<SessionHistory?> GetSessionHistoryAsync(string sessionId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/sessions/{sessionId}/history");
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<SessionHistory>();
                return result;
            }
            else
            {
                var errorMsg = await response.Content.ReadAsStringAsync();
                System.Diagnostics.Debug.WriteLine($"API Error fetching history: {response.StatusCode} - {errorMsg}");
                return null;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error fetching history: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Deletes a session
    /// </summary>
    public async Task<bool> DeleteSessionAsync(string sessionId)
    {
        try
        {
            var response = await _httpClient.DeleteAsync($"{_baseUrl}/api/sessions/{sessionId}");
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<DeleteSessionResponse>();
                if (result != null && result.Success)
                {
                    if (_activeSessionId == sessionId)
                    {
                        _activeSessionId = null;
                    }
                    return true;
                }
                return false;
            }
            else
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                System.Diagnostics.Debug.WriteLine($"API Error deleting session: {response.StatusCode} - {errorContent}");
                return false;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error deleting session: {ex.Message}");
            return false;
        }
    }

    public async Task<ChatResponse> SendFoundationalChatAsync(ChatRequest request)
    {
        try
        {
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/chat/foundational", request);
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<ChatResponse>();
                return result ?? new ChatResponse { Reply = "Error parsing response." };
            }
            else
            {
                var errorContent = await response.Content.ReadAsStringAsync();
                return new ChatResponse 
                { 
                    Reply = $"API Error: {response.StatusCode}. {errorContent}",
                    Blocked = true 
                };
            }
        }
        catch (Exception ex)
        {
            return new ChatResponse 
            { 
                Reply = $"Connection Error: {ex.Message}",
                Blocked = true 
            };
        }
    }

    /// <summary>
    /// Executes a Red Team campaign
    /// </summary>
    public async Task<object?> ExecuteRedTeamCampaignAsync(object campaignConfig)
    {
        try
        {
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/redteam", campaignConfig);
            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadFromJsonAsync<object>();
            }
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error executing redteam campaign: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// Gets evaluation history
    /// </summary>
    public async Task<object?> GetEvaluationHistoryAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/eval");
            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadFromJsonAsync<object>();
            }
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error fetching evaluation history: {ex.Message}");
            return null;
        }
    }
}