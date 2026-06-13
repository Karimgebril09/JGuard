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
    private ChatApiService _chatService;

    public ChatPage()
    {
        this.InitializeComponent();
        _chatService = AppState.Instance.ChatApiService;
        
        // Initialize UI from AppState
        LLMSourceCombo.SelectedIndex = AppState.Instance.LLMSourceType == "OpenSource" ? 0 : 1;
        LLMTypeTextBox.Text = AppState.Instance.LLMType;
        ApiKeyPasswordBox.Password = AppState.Instance.LLMApiKey;
        BaseUrlTextBox.Text = AppState.Instance.ApiBaseUrl;
        
        ObfuscationCheckBox.IsChecked = AppState.Instance.IsObfuscationEnabled;
        MultiTurnCheckBox.IsChecked = AppState.Instance.IsMultiTurnEnabled;
        RoleplayCheckBox.IsChecked = AppState.Instance.IsRoleplayingEnabled;
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
        if (BaseUrlTextBox.Text != AppState.Instance.ApiBaseUrl)
        {
            AppState.Instance.ApiBaseUrl = BaseUrlTextBox.Text;
            _chatService = new ChatApiService(AppState.Instance.ApiBaseUrl);
        }

        // Prepare request
        var request = new ChatRequest
        {
            Prompt = prompt,
            LocalLlm = LLMSourceCombo.SelectedIndex == 0,
            LlmApiKey = ApiKeyPasswordBox.Password,
            LlmType = LLMTypeTextBox.Text,
            ObfuscationProtection = ObfuscationCheckBox.IsChecked ?? false,
            MultiTurnProtection = MultiTurnCheckBox.IsChecked ?? false,
            RoleplayProtection = RoleplayCheckBox.IsChecked ?? false,
            History = ChatMessages
                .Take(ChatMessages.Count - 1) // Exclude the message we just added
                .Select(m => new ChatMessage 
                { 
                    Role = m.Role, 
                    Content = m.Content 
                }).ToList()
        };

        // Scroll to bottom
        ScrollToBottom();

        // Call API
        SendButton.IsEnabled = false;
        try
        {
            var response = await _chatService.SendFoundationalChatAsync(request);
            
            // Parse timestamp securely
            DateTime ts;
            if (!DateTime.TryParse(response.Timestamp, out ts))
            {
                ts = DateTime.Now;
            }

            // Add assistant response to UI
            var assistantMsg = new ChatMessageDisplay
            {
                Content = response.Reply,
                Role = "assistant",
                Timestamp = ts,
                Blocked = response.Blocked,
                TriggeredDefense = response.TriggeredDefense
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