using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using JGuard.Pages;
using JGuard.Services;
using System;
using System.Threading.Tasks;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace JGuard;

public sealed partial class MainWindow : Window
{
    private bool _navigated = false;
    private bool _syncingSelection = false;

    public MainWindow()
    {
        InitializeComponent();

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        AppWindow.SetIcon("Assets/AppIcon.ico");

        NavFrame.Navigated += NavFrame_Navigated;
        this.Activated += MainWindow_Activated;
    }

    // Keep the navigation pane highlight in sync with the frame, including back/forward navigation
    private void NavFrame_Navigated(object sender, Microsoft.UI.Xaml.Navigation.NavigationEventArgs e)
    {
        NavigationViewItem? target =
            e.SourcePageType == typeof(HomePage) ? NavView.MenuItems[0] as NavigationViewItem :
            e.SourcePageType == typeof(RedTeamPage) ? NavView.MenuItems[1] as NavigationViewItem :
            e.SourcePageType == typeof(EvaluationPage) ? NavView.MenuItems[2] as NavigationViewItem :
            null;

        if (target != null && !ReferenceEquals(NavView.SelectedItem, target))
        {
            _syncingSelection = true;
            NavView.SelectedItem = target;
            _syncingSelection = false;
        }
    }

    private async void MainWindow_Activated(object sender, WindowActivatedEventArgs args)
    {
        if (_navigated || args.WindowActivationState == WindowActivationState.Deactivated) return;

        _navigated = true;
        this.Activated -= MainWindow_Activated;

        // Let the content settle before showing a dialog
        await Task.Delay(300);
        await ShowStartupChooserAsync();
    }

    private async Task ShowStartupChooserAsync()
    {
        var startupDialog = new StartupDialog { XamlRoot = this.Content.XamlRoot };
        await startupDialog.ShowAsync();

        switch (startupDialog.Choice)
        {
            case StartupChoice.Home:
                await ShowSessionDialogAsync();
                // If the user cancelled the session dialog, don't leave the frame blank
                if (NavFrame.Content == null) NavView_NavigateToRedTeam();
                break;
            case StartupChoice.Evaluation:
                NavView_NavigateToEvaluation();
                break;
            default:
                // Red Team is the safe default for both the explicit choice and a cancelled chooser
                NavView_NavigateToRedTeam();
                break;
        }
    }

    public async Task ShowSessionDialogAsync()
    {
        try
        {
            var sessionDialog = new SessionDialog();
            sessionDialog.XamlRoot = this.Content.XamlRoot;
            var result = await sessionDialog.ShowAsync();

            if (result == ContentDialogResult.Primary && sessionDialog.SelectedSession != null)
            {
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
                state.IsWebSearchEnabled = config.WebSearchProtection;
                state.IsCodeExecutionEnabled = config.CodeExecutionProtection;
                state.IsRagEnabled = config.RagProtection;
                state.IsEmailEnabled = config.EmailProtection;
                state.IsDocumentEnabled = config.DocumentProtection;

                state.IsConfigurationLocked = true;
                state.ApiService.SetActiveSessionId(sessionDialog.SelectedSession.SessionId);
                state.ActiveSessionIsNew = sessionDialog.IsNewSession;

                NavView_NavigateToHome();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Session dialog error: {ex.Message}");

            var errorDialog = new ContentDialog
            {
                Title = "Session Error",
                Content = $"Failed to initialize session: {ex.Message}",
                CloseButtonText = "OK",
                XamlRoot = this.Content.XamlRoot
            };
            await errorDialog.ShowAsync();
        }
    }

    // The NavFrame.Navigated handler keeps the pane highlight in sync, so these only navigate
    private void NavView_NavigateToHome()
    {
        if (NavFrame.CurrentSourcePageType != typeof(HomePage)) NavFrame.Navigate(typeof(HomePage));
    }

    private void NavView_NavigateToRedTeam()
    {
        if (NavFrame.CurrentSourcePageType != typeof(RedTeamPage)) NavFrame.Navigate(typeof(RedTeamPage));
    }

    private void NavView_NavigateToEvaluation()
    {
        if (NavFrame.CurrentSourcePageType != typeof(EvaluationPage)) NavFrame.Navigate(typeof(EvaluationPage));
    }

    private void TitleBar_PaneToggleRequested(TitleBar sender, object args)
    {
        NavView.IsPaneOpen = !NavView.IsPaneOpen;
    }

    private void TitleBar_BackRequested(TitleBar sender, object args)
    {
        NavFrame.GoBack();
    }

    private async void NavView_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        // Ignore selection changes that originate from syncing the highlight after a back/forward navigation
        if (_syncingSelection) return;

        if (args.SelectedItem is NavigationViewItem item)
        {
            switch (item.Tag)
            {
                case "home":
                    // The chat page requires an active session. If none is set yet,
                    // gate it behind the session dialog instead of opening the chat directly.
                    await NavigateToHomeWithSessionAsync();
                    break;
                case "redteam":
                    if (NavFrame.CurrentSourcePageType != typeof(RedTeamPage)) NavFrame.Navigate(typeof(RedTeamPage));
                    break;
                case "evaluation":
                    if (NavFrame.CurrentSourcePageType != typeof(EvaluationPage)) NavFrame.Navigate(typeof(EvaluationPage));
                    break;
            }
        }
    }

    private async Task NavigateToHomeWithSessionAsync()
    {
        if (NavFrame.CurrentSourcePageType == typeof(HomePage)) return;

        // An active session already exists — go straight to the chat.
        if (!string.IsNullOrEmpty(AppState.Instance.ApiService.GetActiveSessionId))
        {
            NavView_NavigateToHome();
            return;
        }

        // No session yet: require the user to pick/create one. ShowSessionDialogAsync
        // navigates to Home itself when a session is chosen.
        await ShowSessionDialogAsync();

        // Dialog cancelled (still no session): don't open the chat. Restore the nav
        // highlight to whatever page is actually showing.
        if (string.IsNullOrEmpty(AppState.Instance.ApiService.GetActiveSessionId))
        {
            SyncNavSelectionToFrame();
        }
    }

    // Restore the pane highlight to match the page currently in the frame.
    private void SyncNavSelectionToFrame()
    {
        NavigationViewItem? target =
            NavFrame.CurrentSourcePageType == typeof(HomePage) ? NavView.MenuItems[0] as NavigationViewItem :
            NavFrame.CurrentSourcePageType == typeof(RedTeamPage) ? NavView.MenuItems[1] as NavigationViewItem :
            NavFrame.CurrentSourcePageType == typeof(EvaluationPage) ? NavView.MenuItems[2] as NavigationViewItem :
            null;

        if (target != null && !ReferenceEquals(NavView.SelectedItem, target))
        {
            _syncingSelection = true;
            NavView.SelectedItem = target;
            _syncingSelection = false;
        }
    }
}
