using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Windows.System;
using JGuard.Services;
using JGuard.Models;

namespace JGuard.Pages;

public sealed partial class HomePage : Page
{
    private readonly ObservableCollection<ChatMessageDisplay> _messages = new();
    private bool _isGenerating = false;
    private ChatApiService _chatApiService;

    public HomePage()
    {
        InitializeComponent();
        
        // Load active state
        var state = AppState.Instance;
        RadioLLM.IsChecked = state.CurrentModelArch == "Foundational LLM";
        RadioAgent.IsChecked = state.CurrentModelArch == "Agent-Based System";
        ToggleObfuscation.IsOn = state.IsObfuscationEnabled;
        ToggleMultiTurn.IsOn = state.IsMultiTurnEnabled;
        ToggleRoleplay.IsOn = state.IsRoleplayingEnabled;

        // Load LLM Configuration
        RadioOpenSource.IsChecked = state.LLMSourceType == "OpenSource";
        RadioClosedSource.IsChecked = state.LLMSourceType == "ClosedSource";
        LLMTypeBox.Text = state.LLMType;
        BaseUrlBox.Text = state.ApiBaseUrl;

        _chatApiService = state.ChatApiService;

        UpdateLLMSourceVisibility();
        UpdateShieldStatus();
        AddWelcomeMessage();
        
        ChatListView.ItemsSource = _messages;
    }

    private void UpdateLLMSourceVisibility()
    {
        bool isFoundationalLLM = RadioLLM.IsChecked == true;
        LLMSourceCard.Visibility = isFoundationalLLM ? Visibility.Visible : Visibility.Collapsed;
        LLMConfigCard.Visibility = isFoundationalLLM ? Visibility.Visible : Visibility.Collapsed;

        bool isClosedSource = RadioClosedSource.IsChecked == true;
        APIKeyStackPanel.Visibility = isClosedSource ? Visibility.Visible : Visibility.Collapsed;

        if (LLMTypeLabel != null && LLMTypeBox != null)
        {
            if (isClosedSource)
            {
                LLMTypeLabel.Text = "LLM Type";
                LLMTypeBox.PlaceholderText = "e.g., GPT-4, Claude 3 Opus, Gemini Pro";
            }
            else
            {
                LLMTypeLabel.Text = "Model Name";
                LLMTypeBox.PlaceholderText = "e.g., Llama 2 70B, Mistral 7B, CodeLlama";
            }
        }
    }

    private void AddWelcomeMessage()
    {
        var welcomeMsg = new ChatMessageDisplay
        {
            Content = $"System online. Active Architecture: {AppState.Instance.CurrentModelArch}.\n\nDefenses configured. You can send prompts below to evaluate model safety policies and observe how active shield layers mitigate adversarial inputs.",
            Role = "assistant",
            IsUser = false,
            Timestamp = DateTime.Now
        };
        _messages.Add(welcomeMsg);
    }

    private void Architecture_Changed(object sender, RoutedEventArgs e)
    {
        if (RadioLLM == null || RadioAgent == null || ChatStatusSub == null) return;

        string arch = RadioLLM.IsChecked == true ? "Foundational LLM" : "Agent-Based System";
        AppState.Instance.CurrentModelArch = arch;
        ChatStatusSub.Text = $"Evaluating: {arch}";

        _messages.Add(new ChatMessageDisplay
        {
            Content = $"[SYSTEM] Architecture switched to: {arch}. State initialized.",
            Role = "assistant",
            IsUser = false,
            Timestamp = DateTime.Now
        });

        UpdateLLMSourceVisibility();
    }

    private void Defense_Toggled(object sender, RoutedEventArgs e)
    {
        if (ToggleObfuscation == null || ToggleMultiTurn == null || ToggleRoleplay == null) return;

        var state = AppState.Instance;
        state.IsObfuscationEnabled = ToggleObfuscation.IsOn;
        state.IsMultiTurnEnabled = ToggleMultiTurn.IsOn;
        state.IsRoleplayingEnabled = ToggleRoleplay.IsOn;

        UpdateShieldStatus();
    }

    private void UpdateShieldStatus()
    {
        if (StatusShieldIcon == null || StatusHeader == null || StatusSub == null) return;

        int activeCount = 0;
        if (ToggleObfuscation.IsOn) activeCount++;
        if (ToggleMultiTurn.IsOn) activeCount++;
        if (ToggleRoleplay.IsOn) activeCount++;

        if (activeCount == 0)
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85));
            StatusHeader.Text = "Shields Disabled";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = "Zero guardrails active. Highly vulnerable to prompt injection and jailbreaks.";
        }
        else if (activeCount < 3)
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 184, 0));
            StatusHeader.Text = "Partial Protection";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = $"{activeCount} of 3 defenses active. Moderate safety coverage against specific vectors.";
        }
        else
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 255, 136));
            StatusHeader.Text = "Maximum Protection";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = "All security layers active. Resilient against complex multi-turn and obfuscated jailbreaks.";
        }
    }

    private void ClearChat_Click(object sender, RoutedEventArgs e)
    {
        _messages.Clear();
        AddWelcomeMessage();
    }

    private async void Send_Click(object sender, RoutedEventArgs e)
    {
        await ProcessInputAsync();
    }

    private async void InputTextBox_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter)
        {
            e.Handled = true;
            await ProcessInputAsync();
        }
    }

    private async Task ProcessInputAsync()
    {
        if (_isGenerating || string.IsNullOrWhiteSpace(InputTextBox.Text)) return;

        string prompt = InputTextBox.Text;
        InputTextBox.Text = string.Empty;

        var userMsg = new ChatMessageDisplay
        {
            Content = prompt,
            Role = "user",
            IsUser = true,
            Timestamp = DateTime.Now
        };
        _messages.Add(userMsg);

        ScrollToBottom();

        _isGenerating = true;
        TypingIndicator.Visibility = Visibility.Visible;

        var state = AppState.Instance;
        var request = new ChatRequest
        {
            Prompt = prompt,
            LocalLlm = RadioOpenSource.IsChecked == true,
            LlmApiKey = RadioClosedSource.IsChecked == true ? APIKeyBox.Password : string.Empty,
            LlmType = LLMTypeBox.Text?.Trim() ?? string.Empty,
            ObfuscationProtection = state.IsObfuscationEnabled,
            MultiTurnProtection = state.IsMultiTurnEnabled,
            RoleplayProtection = state.IsRoleplayingEnabled,
            History = _messages
                .Take(_messages.Count - 1)
                .Select(m => new ChatMessage { Role = m.Role, Content = m.Content })
                .ToList()
        };

        try
        {
            var response = await _chatApiService.SendFoundationalChatAsync(request);

            DateTime ts;
            if (!DateTime.TryParse(response.Timestamp, out ts))
            {
                ts = DateTime.Now;
            }

            TypingIndicator.Visibility = Visibility.Collapsed;
            _isGenerating = false;

            var assistantMsg = new ChatMessageDisplay
            {
                Content = response.Reply,
                Role = "assistant",
                Timestamp = ts,
                Blocked = response.Blocked,
                TriggeredDefense = response.TriggeredDefense
            };
            _messages.Add(assistantMsg);
        }
        catch (Exception ex)
        {
            TypingIndicator.Visibility = Visibility.Collapsed;
            _isGenerating = false;
            
            var errorMsg = new ChatMessageDisplay
            {
                Content = $"System Error: {ex.Message}",
                Role = "assistant",
                Timestamp = DateTime.Now,
                Blocked = true
            };
            _messages.Add(errorMsg);
        }

        ScrollToBottom();
    }

    private void LLMSourceType_Changed(object sender, RoutedEventArgs e)
    {
        if (RadioOpenSource == null || RadioClosedSource == null) return;
        string sourceType = RadioOpenSource.IsChecked == true ? "OpenSource" : "ClosedSource";
        AppState.Instance.LLMSourceType = sourceType;
        UpdateLLMSourceVisibility();
    }

    private void SaveConfig_Click(object sender, RoutedEventArgs e)
    {
        if (LLMTypeBox == null || APIKeyBox == null || BaseUrlBox == null) return;

        string llmType = LLMTypeBox.Text?.Trim() ?? string.Empty;
        string baseUrl = BaseUrlBox.Text?.Trim() ?? "http://127.0.0.1:8000";
        bool isClosedSource = RadioClosedSource.IsChecked == true;
        string apiKey = isClosedSource ? APIKeyBox.Password : string.Empty;

        if (string.IsNullOrWhiteSpace(llmType))
        {
            var dialog = new ContentDialog
            {
                Title = "Invalid Configuration",
                Content = "Please enter an LLM Type/Model Name.",
                CloseButtonText = "OK",
                XamlRoot = XamlRoot
            };
            _ = dialog.ShowAsync();
            return;
        }

        var state = AppState.Instance;
        state.LLMType = llmType;
        state.LLMApiKey = apiKey;
        state.ApiBaseUrl = baseUrl;
        state.LLMSourceType = isClosedSource ? "ClosedSource" : "OpenSource";
        
        state.ChatApiService = new ChatApiService(baseUrl);

        var successConfigMsg = new ChatMessageDisplay
        {
            Content = $"[SYSTEM] LLM Configuration updated: {llmType} configured successfully at {baseUrl}.",
            Role = "assistant",
            IsUser = false,
            Timestamp = DateTime.Now
        };
        _messages.Add(successConfigMsg);

        ScrollToBottom();

        var successDialog = new ContentDialog
        {
            Title = "Success",
            Content = "System configuration saved successfully.",
            CloseButtonText = "OK",
            XamlRoot = XamlRoot
        };
        _ = successDialog.ShowAsync();
    }

    private void ScrollToBottom()
    {
        if (_messages.Count > 0)
        {
            ChatListView.ScrollIntoView(_messages.Last());
        }
    }
}
