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
    private readonly SessionManager _sessionManager = new();
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
        await _sessionManager.InitializeAsync();
        var sessions = _sessionManager.GetAllSessions();
        
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
        RadioOpenSource.Checked += (s, e) => UpdateApiKeyVisibility();
        RadioClosedSource.Checked += (s, e) => UpdateApiKeyVisibility();
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
    }

    private async void DeleteSessionButton_Click(object sender, RoutedEventArgs e)
    {
        if (SessionsListBox.SelectedItem is Session session)
        {
            try
            {
                // Delete from API first
                var apiService = AppState.Instance.ApiService;
                bool success = await apiService.DeleteSessionAsync(session.SessionId);

                if (success)
                {
                    // Then from local storage
                    await _sessionManager.InitializeAsync();
                    _sessionManager.DeleteSession(session.SessionId);
                    LoadExistingSessions();
                }
                else
                {
                    // Optionally show error to user
                    System.Diagnostics.Debug.WriteLine($"Failed to delete session {session.SessionId} from backend.");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error during session deletion: {ex.Message}");
            }
        }
    }

    private void UpdateApiKeyVisibility()
    {
        bool isClosedSource = RadioClosedSource.IsChecked == true;
        ApiKeyPanel.Visibility = isClosedSource ? Visibility.Visible : Visibility.Collapsed;
    }

    private async void SessionDialog_PrimaryButtonClick(ContentDialog sender, ContentDialogButtonClickEventArgs args)
    {
        var deferral = args.GetDeferral();

        try
        {
            // Show loading state while processing
            SetLoadingState(true);

            // Ensure session manager is initialized before any save/load operations
            await _sessionManager.InitializeAsync();

            // Try to use selected existing session first
            if (SessionsListBox.SelectedItem is Session existingSession)
            {
                SelectedSession = existingSession;
                IsNewSession = false;
                System.Diagnostics.Debug.WriteLine($"Selected existing session: {SelectedSession.SessionId}");
                return;
            }

            // Otherwise create a new session
            IsNewSession = true;

            string llmType = LLMTypeBox.Text?.Trim() ?? string.Empty;
            if (string.IsNullOrEmpty(llmType))
            {
                args.Cancel = true;
                LLMTypeBox.Header = "LLM Model (REQUIRED)";
                LLMTypeBox.PlaceholderText = "PLEASE ENTER MODEL NAME";
                // Optionally show a small teaching tip or similar if available, 
                // but setting Header is a good visual cue.
                return;
            }

            var config = new SessionConfig
            {
                ChatMode = ChatModeCombo.SelectedIndex == 0 ? "foundational" : "agent",
                LocalLlm = RadioOpenSource.IsChecked == true,
                LlmType = llmType,
                LlmApiKey = ApiKeyBox?.Password ?? string.Empty,
                ObfuscationProtection = ObfuscationCheck.IsChecked == true,
                MultiTurnProtection = MultiTurnCheck.IsChecked == true,
                RoleplayProtection = RoleplayCheck.IsChecked == true,
                PiiProtection = PiiCheck.IsChecked == true,
                PiiStrategy = PiiStrategyCombo.SelectedIndex == 1 ? "encrypt" :
                             (PiiStrategyCombo.SelectedIndex == 2 ? "block" : "mask")
            };

            // Call API to create session
            var apiService = AppState.Instance.ApiService;
            var createdSession = await apiService.CreateSessionAsync(config);

            if (createdSession != null)
            {
                SelectedSession = createdSession;
                // Save the new session locally for history
                _sessionManager.SaveSession(SelectedSession);
                System.Diagnostics.Debug.WriteLine($"Created and saved new session from API: {SelectedSession.SessionId}");
            }
            else
            {
                args.Cancel = true;
                ShowErrorMessage("Failed to create session on backend server. Please ensure the backend is running at: " + AppState.Instance.ApiBaseUrl);
                return;
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"CRITICAL: Error in SessionDialog_PrimaryButtonClick: {ex.Message}");
            System.Diagnostics.Debug.WriteLine(ex.StackTrace);
            
            // If we hit an exception, we must cancel to prevent leaving the app in a broken state
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
