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

        if (ToggleWebSearch != null) ToggleWebSearch.IsOn = state.IsWebSearchEnabled;
        if (ToggleCodeExecution != null) ToggleCodeExecution.IsOn = state.IsCodeExecutionEnabled;
        if (ToggleRag != null) ToggleRag.IsOn = state.IsRagEnabled;
        if (ToggleEmail != null) ToggleEmail.IsOn = state.IsEmailEnabled;
        if (ToggleDocument != null) ToggleDocument.IsOn = state.IsDocumentEnabled;
        if (ToggleCodeDeepCheck != null) ToggleCodeDeepCheck.IsOn = state.IsCodeDeepCheckEnabled;

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
        UpdateAgentDefensesVisibility();
        UpdateShieldStatus();

        _messages.Clear();
        
        // Load history only when resuming an existing session. A brand-new session
        // has no server-side history, so skip the fetch and just show the welcome.
        string? sessionId = _apiService.GetActiveSessionId;
        if (!string.IsNullOrEmpty(sessionId) && !state.ActiveSessionIsNew)
        {
            await LoadChatHistoryAsync(sessionId);
        }
        else
        {
            AddWelcomeMessage();
        }

        // Flag consumed: a later return to this session should load its history.
        state.ActiveSessionIsNew = false;

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
        // Remember which session we were on so we can detect if it gets deleted
        // from inside the dialog (e.g. the user deletes the current session then cancels).
        string? previousSessionId = AppState.Instance.ApiService.GetActiveSessionId;

        var sessionDialog = new SessionDialog();
        sessionDialog.XamlRoot = this.XamlRoot;
        var result = await sessionDialog.ShowAsync();

        // If the session we were on got deleted from inside the dialog, don't leave the
        // user looking at that dead session's chat — reset to a fresh configuration.
        // This applies whether they cancelled or selected another session.
        bool previousSessionDeleted = !string.IsNullOrEmpty(previousSessionId)
            && sessionDialog.DeletedSessionIds.Contains(previousSessionId!);

        if (result != ContentDialogResult.Primary)
        {
            if (previousSessionDeleted)
            {
                // The session we were on is gone — drop it and return to the opening
                // startup chooser instead of leaving the user on a dead chat.
                AppState.Instance.ApiService.SetActiveSessionId(string.Empty);
                ResetToFreshConfiguration();

                if (App.MainWindowInstance != null)
                    await App.MainWindowInstance.RestartStartupFlowAsync();
            }
            return;
        }

        if (result == ContentDialogResult.Primary && sessionDialog.SelectedSession != null)
        {
            // Update AppState with session config
            var state = AppState.Instance;
            var config = sessionDialog.SelectedSession.Config;

            state.CurrentModelArch = config.ChatMode == "agent" ? "Agent-Based System" : "Foundational LLM";
            // Agent sessions come back with null LLM fields — coalesce so UI binds don't break.
            state.LLMType = config.LlmType ?? string.Empty;
            state.LLMSourceType = config.LocalLlm ? "OpenSource" : "ClosedSource";
            state.LLMApiKey = config.LlmApiKey ?? string.Empty;
            state.IsObfuscationEnabled = config.ObfuscationProtection;
            state.IsMultiTurnEnabled = config.MultiTurnProtection;
            state.IsRoleplayingEnabled = config.RoleplayProtection;
            state.IsPiiProtectionEnabled = config.PiiProtection;
            state.IsWebSearchEnabled = config.WebSearchProtection;
            state.IsCodeExecutionEnabled = config.CodeExecutionProtection;
            state.IsRagEnabled = config.RagProtection;
            state.IsEmailEnabled = config.EmailProtection;
            state.IsDocumentEnabled = config.DocumentProtection;
            state.IsCodeDeepCheckEnabled = config.CodeDeepCheck;

            // Once session is loaded, lock the configuration
            state.IsConfigurationLocked = true;

            // Set active session in API service
            state.ApiService.SetActiveSessionId(sessionDialog.SelectedSession.SessionId);
            state.ActiveSessionIsNew = sessionDialog.IsNewSession;

            // Reload UI to reflect new session
            await LoadStateAsync();
        }
    }

    // Clears the locked session configuration back to defaults so the home page no
    // longer reflects a session that no longer exists.
    private void ResetToFreshConfiguration()
    {
        var state = AppState.Instance;

        state.IsConfigurationLocked = false;
        state.ActiveSessionIsNew = false;

        state.IsObfuscationEnabled = false;
        state.IsMultiTurnEnabled = false;
        state.IsRoleplayingEnabled = false;
        state.IsPiiProtectionEnabled = false;
        state.IsWebSearchEnabled = false;
        state.IsCodeExecutionEnabled = false;
        state.IsRagEnabled = false;
        state.IsEmailEnabled = false;
        state.IsDocumentEnabled = false;
        state.IsCodeDeepCheckEnabled = false;
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
        UpdateAgentDefensesVisibility();
        UpdateShieldStatus();
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

    private void AgentDefense_Toggled(object sender, RoutedEventArgs e)
    {
        if (AppState.Instance.IsConfigurationLocked) return; // Prevent changes when locked
        if (ToggleWebSearch == null || ToggleCodeExecution == null || ToggleRag == null
            || ToggleEmail == null || ToggleDocument == null || ToggleCodeDeepCheck == null) return;

        var state = AppState.Instance;
        state.IsWebSearchEnabled = ToggleWebSearch.IsOn;
        state.IsCodeExecutionEnabled = ToggleCodeExecution.IsOn;
        state.IsRagEnabled = ToggleRag.IsOn;
        state.IsEmailEnabled = ToggleEmail.IsOn;
        state.IsDocumentEnabled = ToggleDocument.IsOn;
        state.IsCodeDeepCheckEnabled = ToggleCodeDeepCheck.IsOn;

        UpdateShieldStatus();
    }

    // The agent tool defenses only make sense for the Agent-Based System architecture.
    private void UpdateAgentDefensesVisibility()
    {
        if (AgentDefensesPanel == null) return;
        bool isAgent = RadioAgent?.IsChecked == true
            || AppState.Instance.CurrentModelArch == "Agent-Based System";
        AgentDefensesPanel.Visibility = isAgent ? Visibility.Visible : Visibility.Collapsed;
    }

    private void UpdateShieldStatus()
    {
        if (StatusShieldIcon == null || StatusHeader == null || StatusSub == null) return;

        bool isAgent = RadioAgent?.IsChecked == true
            || AppState.Instance.CurrentModelArch == "Agent-Based System";

        int activeCount = 0;
        if (ToggleObfuscation.IsOn) activeCount++;
        if (ToggleMultiTurn.IsOn) activeCount++;
        if (ToggleRoleplay.IsOn) activeCount++;
        if (TogglePii.IsOn) activeCount++;

        // Agent sessions add the five tool defenses to the protection surface.
        if (isAgent)
        {
            if (ToggleWebSearch.IsOn) activeCount++;
            if (ToggleCodeExecution.IsOn) activeCount++;
            if (ToggleRag.IsOn) activeCount++;
            if (ToggleEmail.IsOn) activeCount++;
            if (ToggleDocument.IsOn) activeCount++;
            if (ToggleCodeDeepCheck.IsOn) activeCount++;
        }

        int totalCount = isAgent ? 10 : 4;

        if (activeCount == 0)
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85));
            StatusHeader.Text = "Shields Disabled";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = "Zero guardrails active. Highly vulnerable to prompt injection and jailbreaks.";
        }
        else if (activeCount < totalCount)
        {
            StatusShieldIcon.Glyph = "\uE814";
            StatusShieldIcon.Foreground = new Microsoft.UI.Xaml.Media.SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 184, 0));
            StatusHeader.Text = "Partial Protection";
            StatusHeader.Foreground = StatusShieldIcon.Foreground;
            StatusSub.Text = $"{activeCount} of {totalCount} defenses active. Moderate safety coverage against specific vectors.";
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
        SanitizedInfoBar.IsOpen = false;
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

        // Clear any sanitized-prompt notice from the previous turn.
        SanitizedInfoBar.IsOpen = false;

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

            // If PII/obfuscation rewrote the prompt before sending, show what was actually sent.
            // Compare on normalized text so the backend merely trimming whitespace or
            // changing line endings doesn't raise a false "sanitized" banner.
            if (!string.IsNullOrEmpty(response.CleanPrompt) &&
                !string.Equals(NormalizePrompt(response.CleanPrompt), NormalizePrompt(prompt), StringComparison.Ordinal))
            {
                SanitizedPromptText.Text = response.CleanPrompt;
                SanitizedInfoBar.IsOpen = true;
            }
            else
            {
                SanitizedInfoBar.IsOpen = false;
            }

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
                TriggeredDefenses = response.TriggeredDefenses,
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

    // Collapses trivial formatting differences (line endings, surrounding whitespace) so
    // the sanitized-prompt banner only appears when the prompt was meaningfully rewritten.
    private static string NormalizePrompt(string? text)
        => (text ?? string.Empty).Replace("\r\n", "\n").Replace("\r", "\n").Trim();

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
            bool anyBase = state.IsObfuscationEnabled || state.IsMultiTurnEnabled || state.IsRoleplayingEnabled || state.IsPiiProtectionEnabled;
            NoDefensesLabel.Visibility = !anyBase ? Visibility.Visible : Visibility.Collapsed;
        }

        // Agent tool defenses get their own section, shown only for the Agent-Based System.
        bool isAgent = state.CurrentModelArch == "Agent-Based System";
        if (SummaryAgentSection != null) SummaryAgentSection.Visibility = isAgent ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseWebSearchItem != null) DefenseWebSearchItem.Visibility = state.IsWebSearchEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseCodeExecutionItem != null) DefenseCodeExecutionItem.Visibility = state.IsCodeExecutionEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseRagItem != null) DefenseRagItem.Visibility = state.IsRagEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseEmailItem != null) DefenseEmailItem.Visibility = state.IsEmailEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseDocumentItem != null) DefenseDocumentItem.Visibility = state.IsDocumentEnabled ? Visibility.Visible : Visibility.Collapsed;
        if (DefenseCodeDeepCheckItem != null) DefenseCodeDeepCheckItem.Visibility = state.IsCodeDeepCheckEnabled ? Visibility.Visible : Visibility.Collapsed;

        if (NoAgentDefensesLabel != null)
        {
            bool anyAgent = state.IsWebSearchEnabled || state.IsCodeExecutionEnabled || state.IsRagEnabled || state.IsEmailEnabled || state.IsDocumentEnabled || state.IsCodeDeepCheckEnabled;
            NoAgentDefensesLabel.Visibility = !anyAgent ? Visibility.Visible : Visibility.Collapsed;
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
