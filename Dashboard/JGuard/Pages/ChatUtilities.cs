using System;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace JGuard.Pages;

public class ChatMessageDisplay
{
    public string Content { get; set; } = string.Empty;
    public string Role { get; set; } = string.Empty;
    public string FormattedTime { get; set; } = string.Empty;
    public bool Blocked { get; set; }
    public string? TriggeredDefense { get; set; }
    public string? Decision { get; set; }
    public string? HarmLabel { get; set; }
    
    public string BlockedStatus => Blocked ? "🚫 BLOCKED" : "";
    public Visibility BlockedVisibility => Blocked ? Visibility.Visible : Visibility.Collapsed;
    
    public string TriggeredDefenseStatus => string.IsNullOrEmpty(TriggeredDefense) ? "" : $"⚠️ Defense: {TriggeredDefense}";
    public Visibility DefenseVisibility => !string.IsNullOrEmpty(TriggeredDefense) ? Visibility.Visible : Visibility.Collapsed;

    public string DecisionStatus => string.IsNullOrEmpty(Decision) ? "" : $"⚖️ Decision: {Decision}";
    public Visibility DecisionVisibility => !string.IsNullOrEmpty(Decision) ? Visibility.Visible : Visibility.Collapsed;

    public string HarmLabelStatus => string.IsNullOrEmpty(HarmLabel) ? "" : $"☣️ Harm: {HarmLabel}";
    public Visibility HarmLabelVisibility => !string.IsNullOrEmpty(HarmLabel) ? Visibility.Visible : Visibility.Collapsed;
    
    // For backwards compatibility with HomePage
    public bool IsUser { get; set; }
    
    private DateTime _timestamp = DateTime.Now;
    public DateTime Timestamp 
    { 
        get => _timestamp;
        set 
        { 
            _timestamp = value;
            FormattedTime = value.ToString("hh:mm tt");
        }
    }
}

public class ChatMessageTemplateSelector : DataTemplateSelector
{
    public DataTemplate? UserMessageTemplate { get; set; }
    public DataTemplate? AssistantMessageTemplate { get; set; }
    
    // For backwards compatibility with HomePage
    public DataTemplate? UserTemplate { get; set; }
    public DataTemplate? ModelTemplate { get; set; }

    protected override DataTemplate? SelectTemplateCore(object item, DependencyObject container)
    {
        if (item is ChatMessageDisplay msg)
        {
            // Use newer Role-based system first, fall back to IsUser for backwards compatibility
            if (!string.IsNullOrEmpty(msg.Role))
            {
                return msg.Role == "user" ? UserMessageTemplate : AssistantMessageTemplate;
            }
            else
            {
                return msg.IsUser ? (UserTemplate ?? UserMessageTemplate) : (ModelTemplate ?? AssistantMessageTemplate);
            }
        }
        return base.SelectTemplateCore(item, container);
    }
}
