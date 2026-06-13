using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using JGuard.Models;

namespace JGuard.Services;

public class ChatApiService
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public ChatApiService(string baseUrl)
    {
        _baseUrl = baseUrl.TrimEnd('/');
        _httpClient = new HttpClient();
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
}