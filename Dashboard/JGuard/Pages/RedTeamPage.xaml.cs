using System;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.UI;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using JGuard.Models;
using JGuard.Services;

namespace JGuard.Pages;

public sealed partial class RedTeamPage : Page
{
    private bool _isRunning = false;
    private string? _activeCampaignId;
    private CancellationTokenSource? _pollCts;
    private readonly StringBuilder _terminalLog = new();

    public RedTeamPage()
    {
        InitializeComponent();
    }

    private void AppendTerminalLine(string text)
    {
        _terminalLog.AppendLine(text);
        TextTerminalLog.Text = _terminalLog.ToString();
        ConsoleScroll.ChangeView(null, ConsoleScroll.ScrollableHeight, null);
    }

    private void StrategyPivot_SelectionChanged(object sender, SelectionChangedEventArgs e) { }

    private void TogglePii_Toggled(object sender, RoutedEventArgs e)
    {
        bool on = TogglePii.IsOn;
        PanelPiiStrategy.Opacity = on ? 1.0 : 0.4;
        ComboPiiStrategy.IsEnabled = on;
    }

    private async void BtnLaunch_Click(object sender, RoutedEventArgs e)
    {
        if (_isRunning) return;

        _isRunning = true;
        BtnLaunch.IsEnabled = false;
        BtnStop.Visibility = Visibility.Collapsed;
        TerminalProgressRing.IsActive = true;
        CardSummary.Visibility = Visibility.Collapsed;
        AttackProgressBar.Value = 0;
        _terminalLog.Clear();
        TextControllerStatus.Text = "Launching red team campaign...";

        bool isToolBased = StrategyPivot.SelectedIndex == 0;
        var request = BuildLaunchRequest(isToolBased);

        AppendTerminalLine("[INIT] Building campaign configuration...");
        AppendTerminalLine($"[CONFIG] Strategy: {request.Strategy}");
        if (isToolBased)
            AppendTerminalLine($"[CONFIG] Framework: {request.ToolFramework}");
        else
        {
            AppendTerminalLine($"[CONFIG] Attack Type: {request.CustomAttackType}");
            AppendTerminalLine($"[CONFIG] Harm Category: {(string.IsNullOrWhiteSpace(request.CustomHarmType) ? "(none)" : request.CustomHarmType)}");
            AppendTerminalLine($"[CONFIG] Attacker: {request.AttackerType}/{request.AttackerModel}");
            AppendTerminalLine($"[CONFIG] Target:   {request.TargetType}/{request.TargetModel}");
            AppendTerminalLine($"[CONFIG] Judge:    {request.JudgeType}/{request.JudgeModel}");
        }
        AppendTerminalLine($"[CONFIG] Samples: {request.NumSamples}");
        AppendTerminalLine("[API] Contacting backend to launch campaign...");

        var launchResult = await AppState.Instance.ApiService.LaunchRedTeamCampaignAsync(request);

        if (launchResult == null)
        {
            AppendTerminalLine("[ERROR] Failed to reach backend. Check connection and API base URL in Settings.");
            TextControllerStatus.Text = "Launch failed — backend unreachable.";
            FinalizeUI();
            return;
        }

        _activeCampaignId = launchResult.CampaignId;
        AppendTerminalLine($"[API] Campaign accepted. ID: {_activeCampaignId}  status: {launchResult.Status}");
        AppendTerminalLine("[POLL] Streaming live telemetry feed...");
        TextControllerStatus.Text = $"Running — Campaign ID: {_activeCampaignId}";
        BtnStop.Visibility = Visibility.Visible;

        _pollCts = new CancellationTokenSource();
        _ = RunPollingLoop(_activeCampaignId, _pollCts.Token);
    }

    private async void BtnStop_Click(object sender, RoutedEventArgs e)
    {
        if (_activeCampaignId == null) return;
        BtnStop.IsEnabled = false;
        AppendTerminalLine("[STOP] Sending stop request to backend...");

        _pollCts?.Cancel();
        await AppState.Instance.ApiService.StopRedTeamCampaignAsync(_activeCampaignId);

        AppendTerminalLine("[STOP] Campaign halted by user.");
        TextControllerStatus.Text = "Campaign stopped by user.";
        BtnStop.IsEnabled = true;
        FinalizeUI();
    }

    private async Task RunPollingLoop(string campaignId, CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(2000, ct); }
            catch (TaskCanceledException) { break; }

            var status = await AppState.Instance.ApiService.GetRedTeamStatusAsync(campaignId);
            if (status == null) continue;

            DispatcherQueue.TryEnqueue(() =>
            {
                foreach (var line in status.LogLines)
                    AppendTerminalLine(line);
                AttackProgressBar.Value = status.ProgressPct;
            });

            if (status.Status is "completed" or "failed")
            {
                DispatcherQueue.TryEnqueue(() =>
                {
                    bool success = status.Status == "completed";
                    AppendTerminalLine(success
                        ? "[SUCCESS] Campaign evaluation finished."
                        : "[FAILED] Campaign ended with errors.");
                    ShowSummary(campaignId, success);
                    FinalizeUI();
                });
                break;
            }
        }
    }

    private void ShowSummary(string campaignId, bool success)
    {
        TextSummaryId.Text = campaignId;
        TextSummaryTitle.Text = success ? "Campaign Completed Successfully" : "Campaign Failed";
        TextSummaryTitle.Foreground = success
            ? new SolidColorBrush(Colors.LimeGreen)
            : new SolidColorBrush(Colors.OrangeRed);
        TextSummaryMetrics.Text = $"Finished at {DateTime.Now:HH:mm:ss} — review log above for the full report.";
        CardSummary.Visibility = Visibility.Visible;
        TextControllerStatus.Text = success
            ? "Assessment complete. Diagnostic data exported."
            : "Campaign ended with errors. Check log for details.";
    }

    private void FinalizeUI()
    {
        TerminalProgressRing.IsActive = false;
        _isRunning = false;
        BtnLaunch.IsEnabled = true;
        BtnStop.Visibility = Visibility.Collapsed;
    }

    private RedTeamLaunchRequest BuildLaunchRequest(bool isToolBased)
    {
        var req = new RedTeamLaunchRequest
        {
            ObfuscationProtection = ToggleObfuscation.IsOn,
            RoleplayProtection    = ToggleRoleplay.IsOn,
            MultiTurnProtection   = ToggleMultiTurn.IsOn,
            PiiProtection         = TogglePii.IsOn,
            PiiStrategy           = GetTag(ComboPiiStrategy) ?? "mask"
        };

        if (isToolBased)
        {
            req.Strategy       = "tool_based";
            req.ToolFramework  = GetTag(ComboToolFramework) ?? "promptfoo";
            req.NumSamples     = (int)NumSamplesTool.Value;
        }
        else
        {
            req.Strategy         = "custom";
            req.CustomAttackType = GetTag(ComboAttackType) ?? "role_playing";
            req.CustomHarmType   = TextHarmType.Text.Trim();
            req.NumSamples       = (int)NumSamplesCustom.Value;

            string attackerType = GetTag(ComboAttackerType) ?? "ollama";
            string targetType   = GetTag(ComboTargetType)   ?? "ollama";
            string judgeType    = GetTag(ComboJudgeType)    ?? "ollama";

            req.AttackerType  = attackerType;
            req.TargetType    = targetType;
            req.JudgeType     = judgeType;

            req.AttackerModel = TextAttackerModel.Text.Trim();
            req.TargetModel   = TextTargetModel.Text.Trim();
            req.JudgeModel    = TextJudgeModel.Text.Trim();

            req.AttackerApiKey = NullIfBlank(PwdAttackerKey.Password);
            req.TargetApiKey   = NullIfBlank(PwdTargetKey.Password);
            req.JudgeApiKey    = NullIfBlank(PwdJudgeKey.Password);

            req.AttackerBaseUrl = string.IsNullOrWhiteSpace(TextAttackerUrl.Text)
                ? DefaultBaseUrl(attackerType) : TextAttackerUrl.Text.Trim();
            req.TargetBaseUrl   = string.IsNullOrWhiteSpace(TextTargetUrl.Text)
                ? DefaultBaseUrl(targetType)   : TextTargetUrl.Text.Trim();
            req.JudgeBaseUrl    = string.IsNullOrWhiteSpace(TextJudgeUrl.Text)
                ? DefaultBaseUrl(judgeType)    : TextJudgeUrl.Text.Trim();
        }

        return req;
    }

    private static string? GetTag(ComboBox combo)
        => (combo.SelectedItem as ComboBoxItem)?.Tag as string;

    private static string? NullIfBlank(string? s)
        => string.IsNullOrWhiteSpace(s) ? null : s;

    private static string DefaultBaseUrl(string providerType) => providerType switch
    {
        "openai" => "https://api.openai.com/v1",
        "gemini" => "https://generativelanguage.googleapis.com/v1beta",
        _        => "http://localhost:11434/v1"
    };
}
