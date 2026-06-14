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
    private JGuardApiService? _apiService;

    public HomePage()
    {
        InitializeComponent();
        this.Loaded += HomePage_Loaded;
    }

    private async void HomePage_Loaded(object sender, RoutedEventArgs e)
    {
        this.Loaded -= HomePage_Loaded;
        await LoadStateAsync();
    }

    private async Task LoadStateAsync()
    {
        var state = AppState.Instance;

        // If configuration is already locked, transition the UI immediately before loading values
        if (state.IsConfigurationLocked)
        {
            DisableSettingsControls();
        }
        else
        {
            EnableSettingsControls();
        }
        
        // Load current state into UI
        if (RadioLLM != null) RadioLLM.IsChecked = state.CurrentModelArch == "Foundational LLM";
        if (RadioAgent != null) RadioAgent.IsChecked = state.CurrentModelArch == "Agent-Based System";
        if (ToggleObfuscation != null) ToggleObfuscation.IsOn = state.IsObfuscationEnabled;
        if (ToggleMultiTurn != null) ToggleMultiTurn.IsOn = state.IsMultiTurnEnabled;
        if (ToggleRoleplay != null) ToggleRoleplay.IsOn = state.IsRoleplayingEnabled;
        if (TogglePii != null) TogglePii.IsOn = state.IsPiiProtectionEnabled;

        if (RadioOpenSource != null) RadioOpenSource.IsChecked = state.LLMSourceType == "OpenSource";
        if (RadioClosedSource != null) RadioClosedSource.IsChecked = state.LLMSourceType == "ClosedSource";
        if (LLMTypeBox != null) LLMTypeBox.Text = state.LLMType;
        if (BaseUrlBox != null) BaseUrlBox.Text = state.ApiBaseUrl;
        
        // Load API key if available
        if (!string.IsNullOrEmpty(state.LLMApiKey) && APIKeyBox != null)
        {
            APIKeyBox.Password = state.LLMApiKey;
        }

        _apiService = state.ApiService;
        
        UpdateLLMSourceVisibility();
        UpdateShieldStatus();
        
        _messages.Clear();
        
        // Load history if active session exists
        string? sessionId = _apiService.GetActiveSessionId;
        if (!string.IsNullOrEmpty(sessionId))
        {
            await LoadChatHistoryAsync(sessionId);
        }
        else
        {
            AddWelcomeMessage();
        }

        ChatListView.ItemsSource = _messages;
    }

    private async Task LoadChatHistoryAsync(string sessionId)
    {
        if (_apiService == null) return;

        try
        {
            var history = await _apiService.GetSessionHistoryAsync(sessionId);
            if (history != null && history.History != null && history.History.Any())
            {
                foreach (var msg in history.History)
                {
                    _messages.Add(new ChatMessageDisplay
                    {
                        Content = msg.Content,
                        Role = msg.Role,
                        IsUser = msg.Role.ToLower() == "user",
                        Timestamp = DateTime.Now // Backend doesn't provide individual timestamps yet
                    });
                }
                
                // Scroll to bottom
                if (_messages.Any())
                {
                    ChatListView.ScrollIntoView(_messages.Last());
                }
            }
            else
            {
                AddWelcomeMessage();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Error loading history: {ex.Message}");
            AddWelcomeMessage();
        }
    }

    private void EnableSettingsControls()
    {
        var state = AppState.Instance;
        state.IsConfigurationLocked = false;

        if (ConfigurationPanel != null)
        {
            ConfigurationPanel.Visibility = Visibility.Visible;
            // ConfigurationPanel is a StackPanel, which doesn't have IsEnabled. 
            // We'll set Visibility and Opacity correctly, and child controls will be updated via LoadState/AppState
            ConfigurationPanel.Opacity = 1.0;
        }

        if (ActiveConfigCard != null) ActiveConfigCard.Visibility = Visibility.Collapsed;
        if (SaveConfigButton != null) SaveConfigButton.Visibility = Visibility.Visible;
    }

    private async void NewSession_Click(object sender, RoutedEventArgs e)
    {
        var sessionDialog = new SessionDialog();
        sessionDialog.XamlRoot = this.XamlRoot;
        var result = await sessionDialog.ShowAsync();

        if (result == ContentDialogResult.Primary && sessionDialog.SelectedSession != null)
        {
            // Update AppState with session config
            var state = AppState.Instance;
            var config = sessionDialog.SelectedSession.Config;

            state.CurrentModelArch = config.ChatMode == "agent" ? "Agent-Based System" : "Foundational LLM";
            state.LLMType = config.LlmType;
            state.LLMSourceType = config.LocalLlm ? "OpenSource" : "ClosedSource";
            state.LLMApiKey = config.LlmApiKey;
            state.IsObfuscationEnabled = config.ObfuscationProtection;
            state.IsMultiTurnEnabled = config.MultiTurnProtection;
            state.IsRoleplayingEnabled = config.RoleplayProtection;
            state.IsPiiProtectionEnabled = config.PiiProtection;

            // Once session is loaded, lock the configuration
            state.IsConfigurationLocked = true;

            // Set active session in API service
            state.ApiService.SetActiveSessionId(sessionDialog.SelectedSession.SessionId);

            // Reload UI to reflect new session
            await LoadStateAsync();
        }
    }

    private void UpdateLLMSourceVisibility()
    {
        if (AppState.Instance.IsConfigurationLocked) return; // Don't flip individual card visibilities if locked

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
        if (AppState.Instance.IsConfigurationLocked) return; // Prevent changes when locked
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
        if (AppState.Instance.IsConfigurationLocked) return; // Prevent changes when locked
        if (ToggleObfuscation == null || ToggleMultiTurn == null || ToggleRoleplay == null || TogglePii == null) return;

        var state = AppState.Instance;
        state.IsObfuscationEnabled = ToggleObfuscation.IsOn;
        state.IsMultiTurnEnabled = ToggleMultiTurn.IsOn;
        state.IsRoleplayingEnabled = ToggleRoleplay.IsOn;
        state.IsPiiProtectionEnabled = TogglePii.IsOn;

        UpdateShieldStatus();
    }

    private void UpdateShieldStatus()
    {
        if (StatusShieldIcon == null || StatusHeader == null || StatusSub == null) return;

        int activeCount = 0;
        if (ToggleObfuscation.IsOn) activeCount++;
        if (ToggleMultiTurn.IsOn) activeCount++;
        if (ToggleRoleplay.IsOn) activeCount++;
        if (TogglePii.IsOn) activeCount++;

        if (activeCount == 0)
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85));
            StatusHeader.Text = "Shields Disabled";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = "Zero guardrails active. Highly vulnerable to prompt injection and jailbreaks.";
        }
        else if (activeCount < 4)
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 184, 0));
            StatusHeader.Text = "Partial Protection";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = $"{activeCount} of 4 defenses active. Moderate safety coverage against specific vectors.";
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
            PiiProtection = state.IsPiiProtectionEnabled,
            History = _messages
                .Take(_messages.Count - 1)
                .Select(m => new ChatMessage { Role = m.Role, Content = m.Content })
                .ToList()
        };

        try
        {
            if (_apiService == null) return;
            
            ChatResponse? response;
            string? sessionId = _apiService.GetActiveSessionId;
            
            if (!string.IsNullOrEmpty(sessionId))
            {
                // Use the session-based chat endpoint
                response = await _apiService.SendChatAsync(sessionId, prompt);
            }
            else
            {
                // Fallback to foundational chat endpoint
                response = await _apiService.SendFoundationalChatAsync(request);
            }

            if (response == null) throw new Exception("Failed to receive response from system.");

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
                TriggeredDefense = response.TriggeredDefense,
                Decision = response.Decision,
                HarmLabel = response.HarmLabel
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
        if (AppState.Instance.IsConfigurationLocked) return; // Prevent changes when locked
        if (RadioOpenSource == null || RadioClosedSource == null) return;
        string sourceType = RadioOpenSource.IsChecked == true ? "OpenSource" : "ClosedSource";
        AppState.Instance.LLMSourceType = sourceType;
        UpdateLLMSourceVisibility();
    }

    private void SaveConfig_Click(object sender, RoutedEventArgs e)
    {
        if (AppState.Instance.IsConfigurationLocked) return;

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
        
        // Update local chat service reference with new base URL
        _apiService = new JGuardApiService(baseUrl);

        var successConfigMsg = new ChatMessageDisplay
        {
            Content = $"[SYSTEM] LLM Configuration updated: {llmType} configured successfully at {baseUrl}.",
            Role = "assistant",
            IsUser = false,
            Timestamp = DateTime.Now
        };
        _messages.Add(successConfigMsg);

        ScrollToBottom();

        // Disable all settings controls to make them static
        DisableSettingsControls();

        // Use a simple InfoBar or text update in the chat instead of a ContentDialog 
        // if you want to avoid potential "single dialog" crashes during transition.
        // For now, we keep the dialog but ensure it's the only one by not using them elsewhere.
    }

    private void DisableSettingsControls()
    {
        var state = AppState.Instance;
        state.IsConfigurationLocked = true; // Lock settings globally

        // Ensure we handle current values from state if controls aren't fully prepped
        string arch = state.CurrentModelArch;
        string model = state.LLMType;
        string source = state.LLMSourceType == "OpenSource" ? "Open Source" : "Closed Source";
        string endpoint = state.ApiBaseUrl;

        // Try to get values from UI if possible for latest unsaved changes that are now being locked
        if (RadioLLM != null) arch = RadioLLM.IsChecked == true ? "Foundational LLM" : "Agent-Based System";
        if (LLMTypeBox != null && !string.IsNullOrEmpty(LLMTypeBox.Text)) model = LLMTypeBox.Text;
        if (RadioOpenSource != null) source = RadioOpenSource.IsChecked == true ? "Open Source" : "Closed Source";
        if (BaseUrlBox != null && !string.IsNullOrEmpty(BaseUrlBox.Text)) endpoint = BaseUrlBox.Text;

        // Populate summary card
        if (SummaryArch != null) SummaryArch.Text = arch;
        if (SummaryModel != null) SummaryModel.Text = model;
        if (SummarySource != null) SummarySource.Text = source;
        if (SummaryEndpoint != null) SummaryEndpoint.Text = endpoint;

        // Update defense summary items based on state (since toggles might be disabled/unreliable)
        if (DefenseObfuscationItem != null) DefenseObfuscationItem.Visibility = state.IsObfuscationEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseMultiTurnItem != null) DefenseMultiTurnItem.Visibility = state.IsMultiTurnEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseRoleplayItem != null) DefenseRoleplayItem.Visibility = state.IsRoleplayingEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefensePiiItem != null) DefensePiiItem.Visibility = state.IsPiiProtectionEnabled ? Visibility.Visible : Visibility.Collapsed;

        if (NoDefensesLabel != null)
        {
            NoDefensesLabel.Visibility = (!state.IsObfuscationEnabled && !state.IsMultiTurnEnabled && !state.IsRoleplayingEnabled && !state.IsPiiProtectionEnabled) 
                ? Visibility.Visible : Visibility.Collapsed;
        }

        // ABSOLUTELY HIDE everything in the configuration panel
        if (ConfigurationPanel != null) 
        {
            ConfigurationPanel.Visibility = Visibility.Collapsed;
        }
        
        if (ActiveConfigCard != null) ActiveConfigCard.Visibility = Visibility.Visible;
        
        // Extra precaution: disable all potential entry points
        if (SaveConfigButton != null) SaveConfigButton.Visibility = Visibility.Collapsed;
        if (LLMConfigCard != null) LLMConfigCard.Visibility = Visibility.Collapsed;
        if (LLMSourceCard != null) LLMSourceCard.Visibility = Visibility.Collapsed;
    }

    private void ScrollToBottom()
    {
        if (_messages.Count > 0)
        {
            ChatListView.ScrollIntoView(_messages.Last());
        }
    }
}
