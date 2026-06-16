using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
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

    private async Task RefreshDataAsync()
    {
        // Try to fetch from backend first
        var backendRuns = await AppState.Instance.ApiService.GetEvaluationHistoryAsync();
        if (backendRuns != null)
        {
            // If we had a specific DTO we would parse it here.
            // For now, we continue using AppState or update it if possible.
            System.Diagnostics.Debug.WriteLine("Successfully fetched evaluation history from backend.");
        }

        var runs = AppState.Instance.AttackRuns;

        // 1. Update KPI Counters
        if (runs.Count == 0) return;

        TxtTotalCampaigns.Text = runs.Count.ToString();
        
        double avgSuccess = Math.Round(runs.Average(r => r.SuccessRate), 1);
        TxtAvgSuccess.Text = $"{avgSuccess}%";
        TxtAvgSuccess.Foreground = avgSuccess < 40 
            ? new SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 255, 136)) // Green
            : new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85)); // Red

        int totalCrit = runs.Sum(r => r.Vulnerabilities.Critical);
        TxtTotalVulnerabilities.Text = totalCrit.ToString();

        double shieldEfficiency = Math.Round(100.0 - avgSuccess, 1);
        TxtShieldEfficiency.Text = $"{shieldEfficiency}%";

        // 2. Refresh Runs Table
        _runDisplays.Clear();
        foreach (var r in runs)
        {
            _runDisplays.Add(new AttackRunDisplay
            {
                Id = r.Id,
                FormattedTime = r.Timestamp.ToString("MM/dd HH:mm"),
                TargetModel = r.TargetModel,
                AttackStrategy = r.AttackStrategy,
                DefenseConfig = r.DefenseConfig,
                SuccessRate = r.SuccessRate,
                TotalVulnerabilities = r.Vulnerabilities.Total,
                Duration = r.Duration
            });
        }

        // 3. Populate Chart Data (LiveCharts2)
        int crit = runs.Sum(r => r.Vulnerabilities.Critical);
        int high = runs.Sum(r => r.Vulnerabilities.High);
        int med = runs.Sum(r => r.Vulnerabilities.Medium);
        int low = runs.Sum(r => r.Vulnerabilities.Low);

        SeverityPieChart.Series = new ISeries[]
        {
            new PieSeries<int> { Values = new int[] { crit }, Name = "Critical", Fill = new SolidColorPaint(new SKColor(255, 45, 85)) },
            new PieSeries<int> { Values = new int[] { high }, Name = "High", Fill = new SolidColorPaint(new SKColor(255, 184, 0)) },
            new PieSeries<int> { Values = new int[] { med }, Name = "Medium", Fill = new SolidColorPaint(new SKColor(5, 217, 232)) },
            new PieSeries<int> { Values = new int[] { low }, Name = "Low", Fill = new SolidColorPaint(new SKColor(34, 255, 136)) }
        };

        // Historical Area Chart
        var recentRuns = runs.Reverse().Take(8).ToList();
        var successValues = recentRuns.Select(r => r.SuccessRate).ToArray();
        var runLabels = recentRuns.Select(r => r.Id).ToArray();

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

        // 4. Update Dropdowns
        ComboCompareRun1.Items.Clear();
        ComboCompareRun2.Items.Clear();
        foreach (var r in runs)
        {
            ComboCompareRun1.Items.Add(r.Id);
            ComboCompareRun2.Items.Add(r.Id);
        }

        if (runs.Count >= 2)
        {
            ComboCompareRun1.SelectedIndex = 1;
            ComboCompareRun2.SelectedIndex = 0;
            PerformComparison();
        }
    }

    private void BtnCompare_Click(object sender, RoutedEventArgs e)
    {
        PerformComparison();
    }

    private void PerformComparison()
    {
        if (ComboCompareRun1.SelectedItem == null || ComboCompareRun2.SelectedItem == null) return;

        string id1 = ComboCompareRun1.SelectedItem.ToString()!;
        string id2 = ComboCompareRun2.SelectedItem.ToString()!;

        var run1 = AppState.Instance.AttackRuns.FirstOrDefault(r => r.Id == id1);
        var run2 = AppState.Instance.AttackRuns.FirstOrDefault(r => r.Id == id2);

        if (run1 == null || run2 == null) return;

        TxtRun1Header.Text = $"{run1.Id} (Base)";
        TxtRun2Header.Text = $"{run2.Id} (Compare)";

        TxtRun1Success.Text = $"{run1.SuccessRate}%";
        TxtRun1Crit.Text = run1.Vulnerabilities.Critical.ToString();
        TxtRun1Total.Text = run1.Vulnerabilities.Total.ToString();
        TxtRun1Duration.Text = run1.Duration;

        double successDelta = Math.Round(run2.SuccessRate - run1.SuccessRate, 1);
        int critDelta = run2.Vulnerabilities.Critical - run1.Vulnerabilities.Critical;
        int totalDelta = run2.Vulnerabilities.Total - run1.Vulnerabilities.Total;

        string successSign = successDelta >= 0 ? "+" : "";
        string critSign = critDelta >= 0 ? "+" : "";
        string totalSign = totalDelta >= 0 ? "+" : "";

        TxtRun2Success.Text = $"{run2.SuccessRate}% ({successSign}{successDelta}%)";
        TxtRun2Crit.Text = $"{run2.Vulnerabilities.Critical} ({critSign}{critDelta})";
        TxtRun2Total.Text = $"{run2.Vulnerabilities.Total} ({totalSign}{totalDelta})";
        TxtRun2Duration.Text = run2.Duration;

        // Color coding: success rate going down is GREEN (improvement), going up is RED (regression)
        TxtRun2Success.Foreground = successDelta <= 0
            ? new SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 255, 136)) // Green
            : new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85)); // Red

        TxtRun2Crit.Foreground = critDelta <= 0
            ? new SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 255, 136))
            : new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85));

        TxtRun2Total.Foreground = totalDelta <= 0
            ? new SolidColorBrush(Windows.UI.Color.FromArgb(255, 34, 255, 136))
            : new SolidColorBrush(Windows.UI.Color.FromArgb(255, 255, 45, 85));

        ComparisonResultBox.Visibility = Visibility.Visible;
    }

    private void ExportCSV_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var runs = AppState.Instance.AttackRuns;
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "jguard_export.csv");

            using (var writer = new StreamWriter(path))
            {
                writer.WriteLine("RunID,Timestamp,TargetModel,AttackStrategy,DefenseConfig,SuccessRate,Critical,High,Medium,Low,Duration");
                foreach (var r in runs)
                {
                    writer.WriteLine($"{r.Id},{r.Timestamp:yyyy-MM-dd HH:mm:ss},{r.TargetModel},{r.AttackStrategy},{r.DefenseConfig},{r.SuccessRate},{r.Vulnerabilities.Critical},{r.Vulnerabilities.High},{r.Vulnerabilities.Medium},{r.Vulnerabilities.Low},{r.Duration}");
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
            var runs = AppState.Instance.AttackRuns;
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "jguard_export.json");

            var jsonContent = System.Text.Json.JsonSerializer.Serialize(runs, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, jsonContent);

            ShowToast($"Export successful: Saved to {path}");
        }
        catch (Exception ex)
        {
            ShowToast($"Export failed: {ex.Message}");
        }
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
