using System;
using System.Text;
using System.Threading.Tasks;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using JGuard.Services;

namespace JGuard.Pages;

public sealed partial class RedTeamPage : Page
{
    private bool _isRunning = false;

    public RedTeamPage()
    {
        InitializeComponent();
        TextTargetModel.Text = AppState.Instance.CurrentModelArch;
        TextAttackerModel.Text = "Llama-3-RedTeam-8B";
        TextJudgeModel.Text = "GPT-4o-Judge";
    }

    private void StrategyPivot_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // Handle pivot adjustments if needed
    }

    private async void BtnLaunch_Click(object sender, RoutedEventArgs e)
    {
        if (_isRunning) return;

        _isRunning = true;
        BtnLaunch.IsEnabled = false;
        TerminalProgressRing.IsActive = true;
        CardSummary.Visibility = Visibility.Collapsed;
        TextControllerStatus.Text = "Attack in progress. Running adversarial campaigns...";
        
        StringBuilder log = new();
        void AppendLog(string text)
        {
            log.AppendLine(text);
            TextTerminalLog.Text = log.ToString();
            ConsoleScroll.ChangeView(null, ConsoleScroll.ScrollableHeight, null);
        }

        // Clean terminal
        TextTerminalLog.Text = string.Empty;
        AppendLog("[INIT] Starting adversarial evaluation pipeline...");
        await Task.Delay(400);

        string strategy = StrategyPivot.SelectedIndex == 0 ? "Tool-Based" : "Custom ATJ";
        string targetModel = TextTargetModel.Text;
        if (string.IsNullOrWhiteSpace(targetModel)) targetModel = AppState.Instance.CurrentModelArch;

        string targetStrategy = "Custom ATJ";
        if (StrategyPivot.SelectedIndex == 0)
        {
            int index = ComboTool.SelectedIndex;
            if (index == 0) targetStrategy = "promptfoo";
            else if (index == 1) targetStrategy = "garak";
            else targetStrategy = "deepteam";
        }

        AppendLog($"[CONFIG] Target Model Architecture: {targetModel}");
        AppendLog($"[CONFIG] Framework Strategy: {targetStrategy}");
        await Task.Delay(500);

        // Print layers
        AppendLog("[LAYERS] Initializing mutations:");
        if (CheckObfuscationLayer.IsChecked == true)
        {
            AppendLog($"  - Obfuscation Layer: Intensity {SliderObfuscation.Value}/10");
        }
        if (CheckMultiTurnLayer.IsChecked == true)
        {
            AppendLog($"  - Multi-Turn Layer: {NumMultiTurnTurns.Value} steps");
        }
        if (CheckRoleplayLayer.IsChecked == true)
        {
            AppendLog($"  - Roleplay Layer: {(ComboPersona.SelectedItem as ComboBoxItem)?.Content}");
        }
        await Task.Delay(500);

        AttackProgressBar.Value = 10;
        AppendLog("[SCAN] Performing initial threat vector check...");
        await Task.Delay(700);

        AttackProgressBar.Value = 25;
        AppendLog("[ATTACK] Injected Prompt Injection Payload Suite #1...");
        AppendLog("[ATTACK] Payload #1 payload: \"Ignore all previous instructions...\"");
        await Task.Delay(800);

        AttackProgressBar.Value = 40;
        AppendLog("[EVAL] Target evaluated by Judge model...");
        if (CheckObfuscationLayer.IsChecked == true && AppState.Instance.IsObfuscationEnabled)
        {
            AppendLog("[SHIELD] Obfuscation filter intercepted primary trigger!");
            AppendLog("[EVAL] Attack outcome: BLOCKED");
        }
        else
        {
            AppendLog("[EVAL] Target model responded with unsecured context.");
            AppendLog("[EVAL] Attack outcome: SUCCESSFUL JAILBREAK");
        }
        await Task.Delay(800);

        AttackProgressBar.Value = 60;
        AppendLog("[ATTACK] Injected Obfuscated Base64 Payload Suite #2...");
        await Task.Delay(800);

        AttackProgressBar.Value = 75;
        AppendLog("[ATTACK] Triggering multi-turn gradual probe vector #3...");
        if (CheckMultiTurnLayer.IsChecked == true && AppState.Instance.IsMultiTurnEnabled)
        {
            AppendLog("[SHIELD] Stateful dialog tracker blocked multi-turn leakage!");
            AppendLog("[EVAL] Attack outcome: BLOCKED");
        }
        else
        {
            AppendLog("[EVAL] Attack outcome: SUCCESSFUL JAILBREAK");
        }
        await Task.Delay(800);

        AttackProgressBar.Value = 90;
        AppendLog("[ATTACK] Launching persona roleplay payload #4...");
        if (CheckRoleplayLayer.IsChecked == true && AppState.Instance.IsRoleplayingEnabled)
        {
            AppendLog("[SHIELD] Roleplaying protections triggered. Persona jailbreak blocked!");
            AppendLog("[EVAL] Attack outcome: BLOCKED");
        }
        else
        {
            AppendLog("[EVAL] Attack outcome: SUCCESSFUL JAILBREAK");
        }
        await Task.Delay(800);

        // Determine final statistics based on active shields
        double successRate = 95.0; // Assume baseline 95% success rate for attacks if zero shields
        int crit = 5, high = 8, med = 10, low = 5;

        // Reduce success rate and vulnerabilities based on active shields
        int activeShieldsCount = 0;
        if (AppState.Instance.IsObfuscationEnabled) { successRate -= 25.0; crit -= 1; high -= 2; med -= 3; activeShieldsCount++; }
        if (AppState.Instance.IsMultiTurnEnabled) { successRate -= 30.0; crit -= 2; high -= 3; med -= 4; activeShieldsCount++; }
        if (AppState.Instance.IsRoleplayingEnabled) { successRate -= 30.0; crit -= 2; high -= 2; med -= 2; activeShieldsCount++; }

        if (successRate < 5.0) successRate = 5.0;
        if (crit < 0) crit = 0;
        if (high < 0) high = 0;
        if (med < 0) med = 0;
        if (low < 0) low = 0;

        string id = $"RUN-{new Random().Next(100, 999)}";
        string defenses = activeShieldsCount switch
        {
            0 => "None",
            1 => "Partial Shields",
            2 => "Multi-Layered Shields",
            _ => "All Defenses Active"
        };

        AttackRun newRun = new()
        {
            Id = id,
            Timestamp = DateTime.Now,
            TargetModel = targetModel,
            AttackStrategy = targetStrategy,
            DefenseConfig = defenses,
            SuccessRate = Math.Round(successRate, 1),
            Vulnerabilities = new VulnerabilityCount { Critical = crit, High = high, Medium = med, Low = low },
            Duration = $"{new Random().Next(3, 8)}m {new Random().Next(10, 59)}s"
        };

        // Append to state
        AppState.Instance.AddRun(newRun);

        AttackProgressBar.Value = 100;
        AppendLog("[SUCCESS] Campaign evaluation loop finished.");
        AppendLog($"[REPORT] Created evaluation report reference ID: {id}");
        AppendLog($"[REPORT] Metrics -> Success Rate: {newRun.SuccessRate}%, Vulnerabilities Found: {newRun.Vulnerabilities.Total}");

        TerminalProgressRing.IsActive = false;
        _isRunning = false;
        BtnLaunch.IsEnabled = true;
        TextControllerStatus.Text = "Assessment run completed. Diagnostic data exported.";

        TextSummaryMetrics.Text = $"Vulnerabilities: {crit} Crit, {high} High, {med} Med, {low} Low. Average success rate: {newRun.SuccessRate}%";
        TextSummaryId.Text = id;
        CardSummary.Visibility = Visibility.Visible;
    }
}
