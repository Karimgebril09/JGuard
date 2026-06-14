using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using JGuard.Models;
using JGuard.Services;

namespace JGuard.Pages;

public sealed partial class ChatPage : Page
{
    public ObservableCollection<ChatMessageDisplay> ChatMessages { get; } = new();
    private JGuardApiService _chatService;

    public ChatPage()
    {
        this.InitializeComponent();
        _chatService = AppState.Instance.ApiService;
        
        // Initialize UI from AppState
        LLMSourceCombo.SelectedIndex = AppState.Instance.LLMSourceType == "OpenSource" ? 0 : 1;
        LLMTypeTextBox.Text = AppState.Instance.LLMType;
        ApiKeyPasswordBox.Password = AppState.Instance.LLMApiKey;
        BaseUrlTextBox.Text = AppState.Instance.ApiBaseUrl;
        
        ObfuscationCheckBox.IsChecked = AppState.Instance.IsObfuscationEnabled;
        MultiTurnCheckBox.IsChecked = AppState.Instance.IsMultiTurnEnabled;
        RoleplayCheckBox.IsChecked = AppState.Instance.IsRoleplayingEnabled;

        // Load history if session is active
        this.Loaded += ChatPage_Loaded;
    }

    private async void ChatPage_Loaded(object sender, RoutedEventArgs e)
    {
        string? sessionId = _chatService.GetActiveSessionId;
        if (!string.IsNullOrEmpty(sessionId))
        {
            await LoadChatHistoryAsync(sessionId);
        }
    }

    private async Task LoadChatHistoryAsync(string sessionId)
    {
        try
        {
            var history = await _chatService.GetSessionHistoryAsync(sessionId);
            if (history != null && history.History.Any())
            {
                ChatMessages.Clear();
                foreach (var msg in history.History)
                {
                    ChatMessages.Add(new ChatMessageDisplay
                    {
                        Content = msg.Content,
                        Role = msg.Role,
                        Timestamp = DateTime.Now // API doesn't seem to provide per-message timestamp in history yet
                    });
                }
                ScrollToBottom();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading history: {ex.Message}");
        }
    }

    private async void SendButton_Click(object sender, RoutedEventArgs e)
    {
        await SendMessageAsync();
    }

    private async void PromptTextBox_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == Windows.System.VirtualKey.Enter)
        {
            var shiftKey = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(Windows.System.VirtualKey.Shift);
            if (shiftKey != Windows.UI.Core.CoreVirtualKeyStates.Down)
            {
                e.Handled = true;
                await SendMessageAsync();
            }
        }
    }

    private async Task SendMessageAsync()
    {
        string prompt = PromptTextBox.Text.Trim();
        if (string.IsNullOrEmpty(prompt)) return;

        // Add user message to UI
        var userMsg = new ChatMessageDisplay
        {
            Content = prompt,
            Role = "user",
            Timestamp = DateTime.Now
        };
        ChatMessages.Add(userMsg);
        PromptTextBox.Text = string.Empty;
        
        // Update BaseUrl if changed in UI
        if (BaseUrlTextBox.Text != AppState.Instance.ApiBaseUrl && !string.IsNullOrWhiteSpace(BaseUrlTextBox.Text))
        {
            AppState.Instance.ApiBaseUrl = BaseUrlTextBox.Text;
            var currentSessionId = _chatService.GetActiveSessionId;
            _chatService = new JGuardApiService(AppState.Instance.ApiBaseUrl);
            if (!string.IsNullOrEmpty(currentSessionId))
            {
                _chatService.SetActiveSessionId(currentSessionId);
            }
        }

        // Scroll to bottom
        ScrollToBottom();

        // Call API using session-based endpoint
        SendButton.IsEnabled = false;
        try
        {
            string? sessionId = _chatService.GetActiveSessionId;
            if (string.IsNullOrEmpty(sessionId))
            {
                ChatMessages.Add(new ChatMessageDisplay
                {
                    Content = "Error: No active session. Please select or create a session.",
                    Role = "assistant",
                    Timestamp = DateTime.Now,
                    Blocked = true
                });
                return;
            }

            // Call the session-based chat endpoint
            var response = await _chatService.SendChatAsync(sessionId, prompt);
            
            if (response == null)
            {
                ChatMessages.Add(new ChatMessageDisplay
                {
                    Content = "Error: Failed to get response from API. Check your connection and configuration.",
                    Role = "assistant",
                    Timestamp = DateTime.Now,
                    Blocked = true
                });
                return;
            }

            // Parse timestamp securely
            DateTime ts;
            if (!DateTime.TryParse(response.Timestamp, out ts))
            {
                ts = DateTime.Now;
            }

            var assistantMsg = new ChatMessageDisplay
            {
                Content = response.Reply,
                Role = "assistant",
                Timestamp = ts,
                Blocked = response.Blocked,
                TriggeredDefense = response.TriggeredDefense,
                Decision = response.Decision,
                HarmLabel = response.HarmLabel
            };
            ChatMessages.Add(assistantMsg);
        }
        catch (Exception ex)
        {
            ChatMessages.Add(new ChatMessageDisplay
            {
                Content = $"System Error: {ex.Message}",
                Role = "assistant",
                Timestamp = DateTime.Now,
                Blocked = true
            });
        }
        finally
        {
            SendButton.IsEnabled = true;
            ScrollToBottom();
        }
    }

    private void ScrollToBottom()
    {
        ChatScrollViewer.ChangeView(null, ChatScrollViewer.ScrollableHeight, null);
    }

    private void LLMSourceCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (LLMSourceCombo == null || ApiKeyPasswordBox == null) return;

        if (LLMSourceCombo.SelectedIndex == 0) // Local
        {
            ApiKeyPasswordBox.IsEnabled = false;
        }
        else
        {
            ApiKeyPasswordBox.IsEnabled = true;
        }
    }
}