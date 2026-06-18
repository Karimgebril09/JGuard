using Microsoft.UI.Xaml.Controls;

namespace JGuard.Pages;

public enum StartupChoice
{
    None,
    Home,
    RedTeam,
    Evaluation
}

public sealed partial class StartupDialog : ContentDialog
{
    public StartupChoice Choice { get; private set; } = StartupChoice.None;

    public StartupDialog()
    {
        InitializeComponent();
    }

    private void HomeButton_Click(object sender, Microsoft.UI.Xaml.RoutedEventArgs e)
    {
        Choice = StartupChoice.Home;
        Hide();
    }

    private void RedTeamButton_Click(object sender, Microsoft.UI.Xaml.RoutedEventArgs e)
    {
        Choice = StartupChoice.RedTeam;
        Hide();
    }

    private void EvaluationButton_Click(object sender, Microsoft.UI.Xaml.RoutedEventArgs e)
    {
        Choice = StartupChoice.Evaluation;
        Hide();
    }
}
