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
        _httpClient = new HttpClient
        {
            // LLM inference plus the defense pipeline can take well over the
            // default 100s, especially on cold/local models. Allow up to 10 minutes.
            Timeout = TimeSpan.FromMinutes(10)
        };
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
    /// Gets all sessions from the API
    /// </summary>
    public async Task<List<Session>> GetAllSessionsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/sessions");
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<SessionListResponse>();
                return result?.Sessions ?? new List<Session>();
            }
            System.Diagnostics.Debug.WriteLine($"API Error fetching sessions: {response.StatusCode}");
            return new List<Session>();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error fetching sessions: {ex.Message}");
            return new List<Session>();
        }
    }

    /// <summary>
    /// Creates a new session with the specified configuration
    /// </summary>
    public async Task<Session?> CreateSessionAsync(SessionConfig config)
    {
        try
        {
            // Agent sessions are not backed by a user-supplied LLM, so the source/model
            // fields are irrelevant. Send only the chat mode and protection settings —
            // the backend fills in the rest (local_llm, llm_type, etc.) with defaults.
            object requestBody = config.ChatMode == "agent"
                ? new
                {
                    config = new
                    {
                        chat_mode = config.ChatMode,
                        obfuscation_protection = config.ObfuscationProtection,
                        multi_turn_protection = config.MultiTurnProtection,
                        roleplay_protection = config.RoleplayProtection,
                        pii_protection = config.PiiProtection,
                        pii_strategy = config.PiiStrategy,
                        web_search_protection = config.WebSearchProtection,
                        code_execution_protection = config.CodeExecutionProtection,
                        rag_protection = config.RagProtection,
                        email_protection = config.EmailProtection,
                        document_protection = config.DocumentProtection,
                        code_deep_check = config.CodeDeepCheck
                    }
                }
                : new { config };

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

    public async Task<RedTeamLaunchResponse?> LaunchRedTeamCampaignAsync(RedTeamLaunchRequest request)
    {
        try
        {
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/redteam/launch", request);
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<RedTeamLaunchResponse>();
            System.Diagnostics.Debug.WriteLine($"RedTeam launch error: {response.StatusCode}");
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error launching redteam campaign: {ex.Message}");
            return null;
        }
    }

    public async Task<RedTeamStatusResponse?> GetRedTeamStatusAsync(string campaignId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/redteam/status/{campaignId}");
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<RedTeamStatusResponse>();
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error polling redteam status: {ex.Message}");
            return null;
        }
    }

    public async Task<bool> StopRedTeamCampaignAsync(string campaignId)
    {
        try
        {
            var response = await _httpClient.PostAsync($"{_baseUrl}/api/redteam/stop/{campaignId}", null);
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error stopping redteam campaign: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// GET /api/eval/summary — the top stat cards.
    /// </summary>
    public async Task<EvalSummaryResponse?> GetEvalSummaryAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/eval/summary");
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<EvalSummaryResponse>();
            System.Diagnostics.Debug.WriteLine($"Eval summary error: {response.StatusCode}");
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error fetching eval summary: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// GET /api/eval/attack-trends — historical success rates for the line chart.
    /// </summary>
    public async Task<List<EvalAttackTrend>?> GetEvalAttackTrendsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/eval/attack-trends");
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<List<EvalAttackTrend>>();
            System.Diagnostics.Debug.WriteLine($"Eval attack-trends error: {response.StatusCode}");
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error fetching eval attack-trends: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// GET /api/eval/runs — full run history log for the table.
    /// </summary>
    public async Task<List<EvalRun>?> GetEvalRunsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/eval/runs");
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<List<EvalRun>>();
            System.Diagnostics.Debug.WriteLine($"Eval runs error: {response.StatusCode}");
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error fetching eval runs: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// POST /api/eval/compare — the "Analyze Delta" run comparison engine.
    /// </summary>
    public async Task<EvalCompareResponse?> CompareEvalRunsAsync(string baselineRunId, string compareRunId)
    {
        try
        {
            var body = new EvalCompareRequest
            {
                BaselineRunId = baselineRunId,
                CompareRunId = compareRunId
            };
            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/api/eval/compare", body);
            if (response.IsSuccessStatusCode)
                return await response.Content.ReadFromJsonAsync<EvalCompareResponse>();
            System.Diagnostics.Debug.WriteLine($"Eval compare error: {response.StatusCode}");
            return null;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error comparing eval runs: {ex.Message}");
            return null;
        }
    }
}