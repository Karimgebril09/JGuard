using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using JGuard.Pages;
using JGuard.Services;
using System.Threading.Tasks;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace JGuard;

public sealed partial class MainWindow : Window
{
    private bool _sessionInitialized = false;

    public MainWindow()
    {
        InitializeComponent();

        ExtendsContentIntoTitleBar = true;
        SetTitleBar(AppTitleBar);
        AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        AppWindow.SetIcon("Assets/AppIcon.ico");
        
        this.Activated += MainWindow_Activated;
    }

    private async void MainWindow_Activated(object sender, WindowActivatedEventArgs args)
    {
        if (!_sessionInitialized && args.WindowActivationState != WindowActivationState.Deactivated)
        {
            _sessionInitialized = true;
            this.Activated -= MainWindow_Activated; // Unsubscribe to avoid multiple calls
            
            // Wait a moment for Content to be fully initialized
            await Task.Delay(500);
            await ShowSessionDialogAsync();
        }
    }

    private async Task ShowSessionDialogAsync()
    {
        try
        {
            // Show session dialog - no SessionManager needed, dialog creates session
            var sessionDialog = new SessionDialog();
            sessionDialog.XamlRoot = this.Content.XamlRoot;
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
                var apiService = state.ApiService;
                apiService.SetActiveSessionId(sessionDialog.SelectedSession.SessionId);
                
                // Navigate to home page with session initialized
                NavView_NavigateToHome();
            }
            else
            {
                // User cancelled - exit app
                this.Close();
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Session dialog error: {ex.Message}");
            System.Diagnostics.Debug.WriteLine($"Stack trace: {ex.StackTrace}");
            
            // Show error dialog
            var errorDialog = new ContentDialog
            {
                Title = "Initialization Error",
                Content = $"Failed to initialize sessions: {ex.Message}",
                CloseButtonText = "Exit",
                XamlRoot = this.Content.XamlRoot
            };
            await errorDialog.ShowAsync();
            this.Close();
        }
    }

    private void NavView_NavigateToHome()
    {
        // Manually navigate to home page
        if (NavView.MenuItems.Count > 0 && NavView.MenuItems[0] is NavigationViewItem homeItem)
        {
            NavView.SelectedItem = homeItem;
            NavFrame.Navigate(typeof(HomePage));
        }
    }

    private void TitleBar_PaneToggleRequested(TitleBar sender, object args)
    {
        NavView.IsPaneOpen = !NavView.IsPaneOpen;
    }

    private void TitleBar_BackRequested(TitleBar sender, object args)
    {
        NavFrame.GoBack();
    }

    private void NavView_SelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args)
    {
        if (args.SelectedItem is NavigationViewItem item)
        {
            switch (item.Tag)
            {
                case "home":
                    NavFrame.Navigate(typeof(HomePage));
                    break;
                case "redteam":
                    NavFrame.Navigate(typeof(RedTeamPage));
                    break;
                case "evaluation":
                    NavFrame.Navigate(typeof(EvaluationPage));
                    break;
                default:
                    throw new InvalidOperationException($"Unknown navigation item tag: {item.Tag}");
            }
        }
    }
}
