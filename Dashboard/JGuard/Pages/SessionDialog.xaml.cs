using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using JGuard.Models;
using JGuard.Services;

namespace JGuard.Pages;

public sealed partial class SessionDialog : ContentDialog
{
    public Session? SelectedSession { get; private set; }
    public bool IsNewSession { get; private set; }

    public SessionDialog()
    {
        InitializeComponent();

        // Set defaults
        RadioOpenSource.IsChecked = true;
        ChatModeCombo.SelectedIndex = 0;
        PiiStrategyCombo.SelectedIndex = 0;

        // Load sessions
        LoadExistingSessions();
        SetupEventHandlers();
    }

    private async void LoadExistingSessions()
    {
        var apiService = AppState.Instance.ApiService;
        var sessions = await apiService.GetAllSessionsAsync();

        if (sessions.Any())
        {
            SessionsListBox.ItemsSource = sessions.OrderByDescending(s => s.CreatedAt).ToList();
            SessionsListBox.Visibility = Visibility.Visible;
        }
        else
        {
            SessionsListBox.Visibility = Visibility.Collapsed;
        }
    }

    private void SetupEventHandlers()
    {
        ChatModeCombo.SelectionChanged += (s, e) => UpdateChatModeVisibility();
        RadioOpenSource.Checked += (s, e) => UpdateSourceVisibility();
        RadioClosedSource.Checked += (s, e) => UpdateSourceVisibility();
        UseBaseUrlCheck.Checked += (s, e) => LLMBaseUrlPanel.Visibility = Visibility.Visible;
        UseBaseUrlCheck.Unchecked += (s, e) => LLMBaseUrlPanel.Visibility = Visibility.Collapsed;
        PiiCheck.Checked += (s, e) => PiiStrategyPanel.Visibility = Visibility.Visible;
        PiiCheck.Unchecked += (s, e) => PiiStrategyPanel.Visibility = Visibility.Collapsed;
        
        // If user starts typing a new model name, deselect any existing session
        LLMTypeBox.TextChanged += (s, e) => {
            if (!string.IsNullOrEmpty(LLMTypeBox.Text) && SessionsListBox.SelectedItem != null)
            {
                SessionsListBox.SelectedItem = null;
            }
        };

        SessionsListBox.SelectionChanged += SessionsListBox_SelectionChanged;

        PrimaryButtonClick += SessionDialog_PrimaryButtonClick;

        // Apply the initial chat-mode driven visibility (defaults to Foundational).
        UpdateChatModeVisibility();
    }

    private bool IsAgentMode => ChatModeCombo.SelectedIndex == 1;

    private void UpdateChatModeVisibility()
    {
        bool isAgent = IsAgentMode;

        // Agent sessions have no user-supplied LLM, so hide source/model and their extras.
        LLMSourcePanel.Visibility = isAgent ? Visibility.Collapsed : Visibility.Visible;
        LLMModelPanel.Visibility = isAgent ? Visibility.Collapsed : Visibility.Visible;
        AgentNote.Visibility = isAgent ? Visibility.Visible : Visibility.Collapsed;

        // Email / Web / RAG / Document defenses only apply to agent sessions.
        AgentDefensesPanel.Visibility = isAgent ? Visibility.Visible : Visibility.Collapsed;

        if (isAgent)
        {
            ClosedSourceNote.Visibility = Visibility.Collapsed;
            OpenSourceExtrasPanel.Visibility = Visibility.Collapsed;
            ApiKeyPanel.Visibility = Visibility.Collapsed;

            // Drop any leftover "required" validation styling on the model box.
            LLMTypeBox.Header = "LLM Model";
        }
        else
        {
            // Restore the source-dependent panels for foundational mode.
            UpdateSourceVisibility();
        }
    }

    private async void DeleteSessionButton_Click(object sender, RoutedEventArgs e)
    {
        if (SessionsListBox.SelectedItem is Session session)
        {
            try
            {
                var apiService = AppState.Instance.ApiService;
                bool success = await apiService.DeleteSessionAsync(session.SessionId);

                if (success)
                {
                    LoadExistingSessions();
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"Failed to delete session {session.SessionId} from backend.");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error during session deletion: {ex.Message}");
            }
        }
    }

    private void UpdateSourceVisibility()
    {
        bool isOpen = RadioOpenSource.IsChecked == true;
        OpenSourceExtrasPanel.Visibility = isOpen ? Visibility.Visible : Visibility.Collapsed;
        ApiKeyPanel.Visibility = isOpen ? Visibility.Collapsed : Visibility.Visible;
        ClosedSourceNote.Visibility = isOpen ? Visibility.Collapsed : Visibility.Visible;
        LLMTypeBox.PlaceholderText = isOpen ? "e.g., qwen2.5:3b-instruct" : "e.g., gpt-4o, gemini-3.5-pro";

        // Reset base URL toggle when switching away from open source
        if (!isOpen)
        {
            UseBaseUrlCheck.IsChecked = false;
            LLMBaseUrlPanel.Visibility = Visibility.Collapsed;
        }
    }

    private void ActivateAllDefensesButton_Checked(object sender, RoutedEventArgs e)
    {
        // Checking PiiCheck also reveals the PII strategy panel via its Checked handler
        ObfuscationCheck.IsChecked = true;
        MultiTurnCheck.IsChecked = true;
        RoleplayCheck.IsChecked = true;
        PiiCheck.IsChecked = true;

        ActivateAllText.Text = "All Active";
        var on = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0xEC, 0xFD, 0xF5)); // #ECFDF5
        ActivateAllText.Foreground = on;
        ActivateAllIcon.Foreground = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0xA7, 0xF3, 0xD0)); // #A7F3D0
    }

    private void ActivateAllDefensesButton_Unchecked(object sender, RoutedEventArgs e)
    {
        ObfuscationCheck.IsChecked = false;
        MultiTurnCheck.IsChecked = false;
        RoleplayCheck.IsChecked = false;
        PiiCheck.IsChecked = false;

        ActivateAllText.Text = "Activate All";
        var off = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0x94, 0xA3, 0xB8)); // #94A3B8
        ActivateAllText.Foreground = off;
        ActivateAllIcon.Foreground = off;
    }

    private void ActivateAllAgentButton_Checked(object sender, RoutedEventArgs e)
    {
        WebSearchDefenseToggle.IsChecked = true;
        CodeExecutionDefenseToggle.IsChecked = true;
        RagDefenseToggle.IsChecked = true;
        EmailDefenseToggle.IsChecked = true;
        DocumentDefenseToggle.IsChecked = true;

        ActivateAllAgentText.Text = "All Active";
        ActivateAllAgentText.Foreground = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0xEC, 0xFD, 0xF5)); // #ECFDF5
        ActivateAllAgentIcon.Foreground = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0xA7, 0xF3, 0xD0)); // #A7F3D0
    }

    private void ActivateAllAgentButton_Unchecked(object sender, RoutedEventArgs e)
    {
        WebSearchDefenseToggle.IsChecked = false;
        CodeExecutionDefenseToggle.IsChecked = false;
        RagDefenseToggle.IsChecked = false;
        EmailDefenseToggle.IsChecked = false;
        DocumentDefenseToggle.IsChecked = false;

        ActivateAllAgentText.Text = "Activate All";
        var off = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0x94, 0xA3, 0xB8)); // #94A3B8
        ActivateAllAgentText.Foreground = off;
        ActivateAllAgentIcon.Foreground = off;
    }

    private async void SessionDialog_PrimaryButtonClick(ContentDialog sender, ContentDialogButtonClickEventArgs args)
    {
        var deferral = args.GetDeferral();

        try
        {
            SetLoadingState(true);

            if (SessionsListBox.SelectedItem is Session existingSession)
            {
                SelectedSession = existingSession;
                IsNewSession = false;
                System.Diagnostics.Debug.WriteLine($"Selected existing session: {SelectedSession.SessionId}");
                return;
            }

            IsNewSession = true;

            bool isAgent = IsAgentMode;

            // The LLM model name is only required for foundational sessions; agent
            // sessions are driven entirely by the backend's built-in agent.
            string llmType = LLMTypeBox.Text?.Trim() ?? string.Empty;
            if (!isAgent && string.IsNullOrEmpty(llmType))
            {
                args.Cancel = true;
                LLMTypeBox.Header = "LLM Model (REQUIRED)";
                LLMTypeBox.PlaceholderText = "PLEASE ENTER MODEL NAME";
                return;
            }

            var config = new SessionConfig
            {
                ChatMode = isAgent ? "agent" : "foundational",
                LocalLlm = RadioOpenSource.IsChecked == true,
                LlmType = llmType,
                LlmApiKey = (!isAgent && RadioClosedSource.IsChecked == true) ? (ApiKeyBox?.Password ?? string.Empty) : string.Empty,
                LlmBaseUrl = (!isAgent && RadioOpenSource.IsChecked == true && UseBaseUrlCheck.IsChecked == true) ? (LLMBaseUrlBox.Text?.Trim() ?? string.Empty) : string.Empty,
                ObfuscationProtection = ObfuscationCheck.IsChecked == true,
                MultiTurnProtection = MultiTurnCheck.IsChecked == true,
                RoleplayProtection = RoleplayCheck.IsChecked == true,
                PiiProtection = PiiCheck.IsChecked == true,
                PiiStrategy = (PiiStrategyCombo.SelectedItem as ComboBoxItem)?.Tag as string ?? "mask",

                // Extra agent defenses are only meaningful for agent sessions.
                WebSearchProtection = isAgent && WebSearchDefenseToggle.IsChecked == true,
                CodeExecutionProtection = isAgent && CodeExecutionDefenseToggle.IsChecked == true,
                RagProtection = isAgent && RagDefenseToggle.IsChecked == true,
                EmailProtection = isAgent && EmailDefenseToggle.IsChecked == true,
                DocumentProtection = isAgent && DocumentDefenseToggle.IsChecked == true
            };

            var apiService = AppState.Instance.ApiService;
            var createdSession = await apiService.CreateSessionAsync(config);

            if (createdSession != null)
            {
                SelectedSession = createdSession;
                System.Diagnostics.Debug.WriteLine($"Created new session from API: {SelectedSession.SessionId}");
            }
            else
            {
                args.Cancel = true;
                ShowErrorMessage("Failed to create session on backend server. Please ensure the backend is running at: " + AppState.Instance.ApiBaseUrl);
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CRITICAL: Error in SessionDialog_PrimaryButtonClick: {ex.Message}");
            System.Diagnostics.Debug.WriteLine(ex.StackTrace);
            args.Cancel = true;
        }
        finally
        {
            SetLoadingState(false);
            deferral.Complete();
        }
    }

    private void SetLoadingState(bool isLoading)
    {
        MainContent.Visibility = isLoading ? Visibility.Collapsed : Visibility.Visible;
        LoadingOverlay.Visibility = isLoading ? Visibility.Visible : Visibility.Collapsed;
        IsPrimaryButtonEnabled = !isLoading;
        IsSecondaryButtonEnabled = !isLoading;
    }

    private void ShowErrorMessage(string message)
    {
        var dialog = new ContentDialog
        {
            Title = "Validation Error",
            Content = message,
            CloseButtonText = "OK",
            RequestedTheme = ElementTheme.Dark,
            XamlRoot = this.XamlRoot
        };
        _ = dialog.ShowAsync();
    }

    private void SessionsListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        DeleteSessionButton.Visibility = SessionsListBox.SelectedItem != null ? Visibility.Visible : Visibility.Collapsed;

        // Revert visuals for items that were deselected
        foreach (var removed in e.RemovedItems)
        {
            if (SessionsListBox.ContainerFromItem(removed) is ListBoxItem removedContainer)
            {
                var border = FindDescendant<Border>(removedContainer);
                if (border != null)
                {
                    border.BorderBrush = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0x33, 0x41, 0x55)); // #334155
                    border.Background = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0x1E, 0x29, 0x3B)); // #1E293B
                }
            }
        }

        // Apply highlight visuals for newly selected items
        foreach (var added in e.AddedItems)
        {
            if (SessionsListBox.ContainerFromItem(added) is ListBoxItem addedContainer)
            {
                var border = FindDescendant<Border>(addedContainer);
                if (border != null)
                {
                    border.BorderBrush = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0x05, 0xD9, 0xE8)); // #05D9E8
                    border.Background = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 0x0F, 0x17, 0x2A)); // #0F172A
                }
            }
        }

        if (SessionsListBox.SelectedItem != null)
        {
            // Clear the new session fields if an existing one is selected
            LLMTypeBox.Text = string.Empty;
            LLMTypeBox.Header = "LLM Model";
        }
    }

    private T? FindDescendant<T>(DependencyObject parent) where T : DependencyObject
    {
        if (parent == null) return null;
        int count = VisualTreeHelper.GetChildrenCount(parent);
        for (int i = 0; i < count; i++)
        {
            var child = VisualTreeHelper.GetChild(parent, i);
            if (child is T t) return t;
            var result = FindDescendant<T>(child);
            if (result != null) return result;
        }
        return null;
    }
}
