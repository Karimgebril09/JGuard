using System;
using System.IO;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;

namespace JGuard;

/// <summary>
/// Provides application-specific behavior to supplement the default Application class.
/// </summary>
public partial class App : Application
{
    private Window? _window;
    
    /// <summary>
    /// Initializes the singleton application object.  This is the first line of authored code
    /// executed, and as such is the logical equivalent of main() or WinMain().
    /// </summary>
    public App()
    {
        InitializeComponent();
        this.UnhandledException += App_UnhandledException;
    }

    private void App_UnhandledException(object sender, Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        try
        {
            string logPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "crash_log.txt");
            File.WriteAllText(logPath, $"UnhandledException:\n{e.Exception}\n\nInner Exception:\n{e.Exception?.InnerException}\n\nMessage: {e.Message}");
            
            // Also write to workspace directory so it's easy to read
            File.WriteAllText("E:\\CMP\\4\\Dashboard\\crash_log.txt", $"UnhandledException:\n{e.Exception}\n\nInner Exception:\n{e.Exception?.InnerException}\n\nMessage: {e.Message}");
        }
        catch { }
    }

    /// <summary>
    /// Invoked when the application is launched.
    /// </summary>
    /// <param name="args">Details about the launch request and process.</param>
    protected override void OnLaunched(Microsoft.UI.Xaml.LaunchActivatedEventArgs args)
    {
        try
        {
            _window = new MainWindow();
            _window.Activate();
        }
        catch (Exception ex)
        {
            try
            {
                File.WriteAllText("E:\\CMP\\4\\Dashboard\\crash_log.txt", $"OnLaunched Crash:\n{ex}");
            }
            catch { }
            throw;
        }
    }
}
