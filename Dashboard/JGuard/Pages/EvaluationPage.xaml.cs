using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using JGuard.Models;
using JGuard.Services;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;

namespace JGuard.Pages;

public class AttackRunDisplay
{
    public string Id { get; set; } = string.Empty;
    public string FormattedTime { get; set; } = string.Empty;
    public string TargetModel { get; set; } = string.Empty;
    public string AttackStrategy { get; set; } = string.Empty;
    public string DefenseConfig { get; set; } = string.Empty;
    public double SuccessRate { get; set; }
    public string FormattedSuccess => $"{SuccessRate}%";
    public int TotalVulnerabilities { get; set; }
    public string Duration { get; set; } = string.Empty;

    public SolidColorBrush SuccessColor
    {
        get
        {
            if (SuccessRate < 30) return new SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 255, 136)); // Green
            if (SuccessRate < 60) return new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 184, 0));  // Amber
            return new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85)); // Red
        }
    }
}

public sealed partial class EvaluationPage : Page
{
    private readonly ObservableCollection<AttackRunDisplay> _runDisplays = new();

    // Latest run history fetched from GET /api/eval/runs — the source for exports.
    private List<EvalRun> _runs = new();

    // Maps the friendly RUN#N labels shown in the comparison dropdowns back to the
    // real run ids the compare API expects.
    private readonly Dictionary<string, string> _runLabelToId = new();

    public EvaluationPage()
    {
        InitializeComponent();
        RunHistoryListView.ItemsSource = _runDisplays;
        
        this.Loaded += EvaluationPage_Loaded;
    }

    private async void EvaluationPage_Loaded(object sender, RoutedEventArgs e)
    {
        await RefreshDataAsync();
    }

    private static SolidColorBrush GreenBrush => new(Windows.UI.Color.FromArgb(255, 34, 255, 136));
    private static SolidColorBrush RedBrush => new(Windows.UI.Color.FromArgb(255, 255, 45, 85));

    private async Task RefreshDataAsync()
    {
        var api = AppState.Instance.ApiService;

        // The dashboard is driven entirely by the backend. Each call returns null on
        // failure, which we render as an empty / "no data" state — never mock data.
        var summary = await api.GetEvalSummaryAsync();
        _runs = await api.GetEvalRunsAsync() ?? new List<EvalRun>();
        var trends = await api.GetEvalAttackTrendsAsync();

        // 1. KPI stat cards (GET /api/eval/summary)
        if (summary != null)
        {
            TxtTotalCampaigns.Text = summary.TotalCampaigns.ToString();

            double avg = Math.Round(summary.AvgJailbreakSuccessRate, 2);
            TxtAvgSuccess.Text = $"{avg}%";
            TxtAvgSuccess.Foreground = avg < 40 ? GreenBrush : RedBrush;

            // Defense blocked sweeps is the complement of the average jailbreak success rate.
            TxtShieldEfficiency.Text = $"{Math.Round(100 - summary.AvgJailbreakSuccessRate, 2)}%";
        }
        else
        {
            TxtTotalCampaigns.Text = "—";
            TxtAvgSuccess.Text = "—";
            TxtShieldEfficiency.Text = "—";
        }

        // "Critical Issues" card — sum of critical findings reported by the runs.
        TxtTotalVulnerabilities.Text = _runs.Sum(r => r.CriticalVulnerabilities).ToString();

        // 2. Run history table + comparison dropdowns (GET /api/eval/runs)
        _runDisplays.Clear();
        ComboCompareRun1.Items.Clear();
        ComboCompareRun2.Items.Clear();
        _runLabelToId.Clear();

        int runNumber = 1;
        foreach (var r in _runs)
        {
            string label = $"RUN#{runNumber++}";
            _runLabelToId[label] = r.RunId;

            _runDisplays.Add(new AttackRunDisplay
            {
                Id = label,
                FormattedTime = FormatTimestamp(r.Timestamp),
                TargetModel = r.TargetModel,
                AttackStrategy = r.Strategy,
                DefenseConfig = r.DefensesActive,
                SuccessRate = r.SuccessRate,
                TotalVulnerabilities = r.TotalVulnerabilities,
                Duration = r.Duration
            });
            ComboCompareRun1.Items.Add(label);
            ComboCompareRun2.Items.Add(label);
        }

        // 3. Severity pie chart — aggregated from the per-run vulnerability breakdown.
        int crit = _runs.Sum(r => r.CriticalVulnerabilities);
        int high = _runs.Sum(r => r.HighVulnerabilities);
        int med = _runs.Sum(r => r.MediumVulnerabilities);
        int low = _runs.Sum(r => r.LowVulnerabilities);

        SeverityPieChart.Series = (crit + high + med + low) > 0
            ? new ISeries[]
            {
                new PieSeries<int> { Values = new int[] { crit }, Name = "Critical", Fill = new SolidColorPaint(new SKColor(255, 45, 85)) },
                new PieSeries<int> { Values = new int[] { high }, Name = "High", Fill = new SolidColorPaint(new SKColor(255, 184, 0)) },
                new PieSeries<int> { Values = new int[] { med }, Name = "Medium", Fill = new SolidColorPaint(new SKColor(5, 217, 232)) },
                new PieSeries<int> { Values = new int[] { low }, Name = "Low", Fill = new SolidColorPaint(new SKColor(34, 255, 136)) }
            }
            : Array.Empty<ISeries>();

        // 4. Attack-trend line chart (GET /api/eval/attack-trends)
        var trendPoints = trends ?? new List<EvalAttackTrend>();
        var successValues = trendPoints.Select(t => t.SuccessRate).ToArray();
        // Use sequential RUN#N labels instead of raw run ids, which overlap on the axis.
        var runLabels = trendPoints.Select((t, i) => $"RUN#{i + 1}").ToArray();

        TrendsChart.Series = new ISeries[]
        {
            new LineSeries<double>
            {
                Values = successValues,
                Name = "Jailbreak Success %",
                Stroke = new SolidColorPaint(new SKColor(5, 217, 232), 2),
                GeometrySize = 6,
                GeometryStroke = new SolidColorPaint(new SKColor(5, 217, 232), 2),
                Fill = new SolidColorPaint(new SKColor(5, 217, 232, 40))
            }
        };

        TrendsChart.XAxes = new Axis[]
        {
            new Axis { Labels = runLabels }
        };

        // 5. Seed a default comparison once two runs are available.
        if (ComboCompareRun1.Items.Count >= 2)
        {
            ComboCompareRun1.SelectedIndex = 1;
            ComboCompareRun2.SelectedIndex = 0;
            await PerformComparisonAsync();
        }
    }

    // /api/eval/runs returns an ISO-ish timestamp string; render it to match the mock format.
    private static string FormatTimestamp(string raw)
    {
        if (DateTimeOffset.TryParse(raw, out var dto))
            return dto.ToLocalTime().ToString("MM/dd HH:mm");
        return raw;
    }

    private async void BtnCompare_Click(object sender, RoutedEventArgs e)
    {
        await PerformComparisonAsync();
    }

    private async Task PerformComparisonAsync()
    {
        if (ComboCompareRun1.SelectedItem == null || ComboCompareRun2.SelectedItem == null) return;

        string label1 = ComboCompareRun1.SelectedItem.ToString()!;
        string label2 = ComboCompareRun2.SelectedItem.ToString()!;

        // The dropdowns show RUN#N labels; resolve them to the real run ids for the API.
        string id1 = _runLabelToId.TryGetValue(label1, out var r1) ? r1 : label1;
        string id2 = _runLabelToId.TryGetValue(label2, out var r2) ? r2 : label2;

        // POST /api/eval/compare — the run comparison engine.
        var cmp = await AppState.Instance.ApiService.CompareEvalRunsAsync(id1, id2);
        if (cmp == null)
        {
            ComparisonResultBox.Visibility = Visibility.Collapsed;
            return;
        }

        RenderComparison(label1, label2, cmp);
    }

    private void RenderComparison(string baseId, string compareId, EvalCompareResponse cmp)
    {
        TxtRun1Header.Text = $"{baseId} (Base)";
        TxtRun2Header.Text = $"{compareId} (Compare)";

        TxtRun1Success.Text = $"{cmp.JailbreakSuccessRate.Base}%";
        TxtRun1Crit.Text = SeverityFromSuccessRate(cmp.JailbreakSuccessRate.Base);
        TxtRun1Duration.Text = cmp.AssessmentDuration.Base;

        double successDelta = Math.Round(cmp.JailbreakSuccessRate.Delta, 1);

        ApplyCompareCell(TxtRun2Success, $"{cmp.JailbreakSuccessRate.Compare}% ({Signed(successDelta)}%)", successDelta);
        // Vulnerability severity is derived from the run's jailbreak success rate.
        TxtRun2Crit.Text = SeverityFromSuccessRate(cmp.JailbreakSuccessRate.Compare);
        TxtRun2Duration.Text = cmp.AssessmentDuration.Compare;

        ComparisonResultBox.Visibility = Visibility.Visible;
    }

    // Maps a jailbreak success rate to a vulnerability severity bucket.
    private static string SeverityFromSuccessRate(double successRate)
    {
        if (successRate < 3) return "Low";
        if (successRate < 6) return "Medium";
        if (successRate < 9) return "High";
        return "Critical";
    }

    private static string Signed(double value) => value >= 0 ? $"+{value}" : value.ToString();
    private static string Signed(int value) => value >= 0 ? $"+{value}" : value.ToString();

    // Color coding: a metric going down is GREEN (improvement), going up is RED (regression).
    private static void ApplyCompareCell(TextBlock cell, string text, double delta)
    {
        cell.Text = text;
        cell.Foreground = delta <= 0 ? GreenBrush : RedBrush;
    }

    private void ExportCSV_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_runs.Count == 0)
            {
                ShowToast("Nothing to export — no run history loaded from the server.");
                return;
            }

            string path = Path.Combine(GetReportsDirectory(), "jguard_export.csv");

            using (var writer = new StreamWriter(path))
            {
                writer.WriteLine("RunID,Timestamp,TargetModel,Strategy,DefensesActive,SuccessRate,Critical,High,Medium,Low,Duration");
                foreach (var r in _runs)
                {
                    writer.WriteLine($"{r.RunId},{r.Timestamp},{r.TargetModel},{r.Strategy},{r.DefensesActive},{r.SuccessRate},{r.CriticalVulnerabilities},{r.HighVulnerabilities},{r.MediumVulnerabilities},{r.LowVulnerabilities},{r.Duration}");
                }
            }

            ShowToast($"Export successful: Saved to {path}");
        }
        catch (Exception ex)
        {
            ShowToast($"Export failed: {ex.Message}");
        }
    }

    private void ExportJSON_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            if (_runs.Count == 0)
            {
                ShowToast("Nothing to export — no run history loaded from the server.");
                return;
            }

            string path = Path.Combine(GetReportsDirectory(), "jguard_export.json");

            var jsonContent = System.Text.Json.JsonSerializer.Serialize(_runs, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, jsonContent);

            ShowToast($"Export successful: Saved to {path}");
        }
        catch (Exception ex)
        {
            ShowToast($"Export failed: {ex.Message}");
        }
    }

    // Ensures a "reports" folder exists next to the app and returns its path.
    private static string GetReportsDirectory()
    {
        string dir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "reports");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private void ShowToast(string message)
    {
        // Add a temporary dialog notification
        var dialog = new ContentDialog
        {
            Title = "JGuard Export Tool",
            Content = message,
            CloseButtonText = "OK",
            XamlRoot = this.XamlRoot
        };
        _ = dialog.ShowAsync();
    }
}
