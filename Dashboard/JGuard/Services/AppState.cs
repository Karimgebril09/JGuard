namespace JGuard.Services;

public class AppState
{
    private static AppState? _instance;
    public static AppState Instance => _instance ??= new AppState();

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

    // True when the active session was just created, so there is no server-side
    // history to fetch. Consumed (and reset) by HomePage when it loads.
    public bool ActiveSessionIsNew { get; set; } = false;

    // API Configuration
    public string ApiBaseUrl { get; set; } = "http://127.0.0.1:8000"; // Default API endpoint
    public JGuardApiService ApiService { get; private set; }

    private AppState()
    {
        ApiService = new JGuardApiService(ApiBaseUrl);
    }
}
