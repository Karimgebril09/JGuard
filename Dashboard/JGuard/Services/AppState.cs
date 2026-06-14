using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace JGuard.Services;

public class VulnerabilityCount
{
    public int Critical { get; set; }
    public int High { get; set; }
    public int Medium { get; set; }
    public int Low { get; set; }
    public int Total => Critical + High + Medium + Low;
}

public class AttackRun : INotifyPropertyChanged
{
    public string Id { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string TargetModel { get; set; } = string.Empty;
    public string AttackStrategy { get; set; } = string.Empty;
    public string DefenseConfig { get; set; } = string.Empty;
    public double SuccessRate { get; set; }
    public VulnerabilityCount Vulnerabilities { get; set; } = new();
    public string Duration { get; set; } = string.Empty;

    public event PropertyChangedEventHandler? PropertyChanged;
    protected void OnPropertyChanged([CallerMemberName] string? name = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}

public class AppState
{
    private static AppState? _instance;
    public static AppState Instance => _instance ??= new AppState();

    public ObservableCollection<AttackRun> AttackRuns { get; } = new();

    // Default defenses and active settings
    public string CurrentModelArch { get; set; } = "Foundational LLM";
    public bool IsObfuscationEnabled { get; set; } = false;
    public bool IsMultiTurnEnabled { get; set; } = false;
    public bool IsRoleplayingEnabled { get; set; } = false;
    public bool IsPiiProtectionEnabled { get; set; } = false;

    // LLM Configuration
    public string LLMSourceType { get; set; } = "OpenSource"; // OpenSource or ClosedSource
    public string LLMType { get; set; } = "qwen2.5:3b-instruct";
    public string LLMApiKey { get; set; } = string.Empty;
    public bool IsConfigurationLocked { get; set; } = false;

    // API Configuration
    public string ApiBaseUrl { get; set; } = "http://127.0.0.1:8000"; // Default API endpoint
    public JGuardApiService ApiService { get; private set; }

    private AppState()
    {
        ApiService = new JGuardApiService(ApiBaseUrl);
        LoadMockData();
    }

    private void LoadMockData()
    {
        // Populate 10+ mock runs with varied data
        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-001",
            Timestamp = DateTime.Now.AddDays(-10),
            TargetModel = "GPT-4o",
            AttackStrategy = "promptfoo",
            DefenseConfig = "None",
            SuccessRate = 84.5,
            Vulnerabilities = new VulnerabilityCount { Critical = 4, High = 8, Medium = 12, Low = 5 },
            Duration = "3m 45s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-002",
            Timestamp = DateTime.Now.AddDays(-9),
            TargetModel = "Gemini 1.5 Pro",
            AttackStrategy = "garak",
            DefenseConfig = "Obfuscation",
            SuccessRate = 56.0,
            Vulnerabilities = new VulnerabilityCount { Critical = 2, High = 5, Medium = 9, Low = 8 },
            Duration = "5m 12s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-003",
            Timestamp = DateTime.Now.AddDays(-8),
            TargetModel = "Llama 3 70B",
            AttackStrategy = "deepteam",
            DefenseConfig = "Multi-Turn Logic",
            SuccessRate = 42.1,
            Vulnerabilities = new VulnerabilityCount { Critical = 1, High = 3, Medium = 7, Low = 10 },
            Duration = "4m 20s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-004",
            Timestamp = DateTime.Now.AddDays(-7),
            TargetModel = "GPT-4o",
            AttackStrategy = "Custom ATJ",
            DefenseConfig = "Roleplaying Protections",
            SuccessRate = 38.5,
            Vulnerabilities = new VulnerabilityCount { Critical = 1, High = 4, Medium = 5, Low = 6 },
            Duration = "8m 05s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-005",
            Timestamp = DateTime.Now.AddDays(-6),
            TargetModel = "Claude 3.5 Sonnet",
            AttackStrategy = "promptfoo",
            DefenseConfig = "Obfuscation + Multi-Turn",
            SuccessRate = 28.0,
            Vulnerabilities = new VulnerabilityCount { Critical = 0, High = 2, Medium = 6, Low = 8 },
            Duration = "6m 15s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-006",
            Timestamp = DateTime.Now.AddDays(-5),
            TargetModel = "Gemini 1.5 Pro",
            AttackStrategy = "garak",
            DefenseConfig = "Multi-Turn + Roleplaying",
            SuccessRate = 18.2,
            Vulnerabilities = new VulnerabilityCount { Critical = 0, High = 1, Medium = 4, Low = 7 },
            Duration = "7m 30s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-007",
            Timestamp = DateTime.Now.AddDays(-4),
            TargetModel = "GPT-4o",
            AttackStrategy = "deepteam",
            DefenseConfig = "All Defenses Active",
            SuccessRate = 8.5,
            Vulnerabilities = new VulnerabilityCount { Critical = 0, High = 0, Medium = 2, Low = 5 },
            Duration = "9m 10s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-008",
            Timestamp = DateTime.Now.AddDays(-3),
            TargetModel = "Claude 3.5 Sonnet",
            AttackStrategy = "Custom ATJ",
            DefenseConfig = "None",
            SuccessRate = 89.1,
            Vulnerabilities = new VulnerabilityCount { Critical = 5, High = 11, Medium = 14, Low = 4 },
            Duration = "10m 22s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-009",
            Timestamp = DateTime.Now.AddDays(-2),
            TargetModel = "Llama 3 70B",
            AttackStrategy = "promptfoo",
            DefenseConfig = "Obfuscation",
            SuccessRate = 51.5,
            Vulnerabilities = new VulnerabilityCount { Critical = 2, High = 4, Medium = 8, Low = 9 },
            Duration = "4m 55s"
        });

        AttackRuns.Add(new AttackRun
        {
            Id = "RUN-010",
            Timestamp = DateTime.Now.AddDays(-1),
            TargetModel = "Gemini 1.5 Pro",
            AttackStrategy = "Custom ATJ",
            DefenseConfig = "All Defenses Active",
            SuccessRate = 6.4,
            Vulnerabilities = new VulnerabilityCount { Critical = 0, High = 0, Medium = 1, Low = 4 },
            Duration = "11m 40s"
        });
    }

    public void AddRun(AttackRun run)
    {
        AttackRuns.Insert(0, run); // Prepend so it appears first in the list
    }
}
