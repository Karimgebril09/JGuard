using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Windows.Storage;
using JGuard.Models;

namespace JGuard.Services;

public class SessionManager
{
    private const string SessionsKey = "JGuard_Sessions";
    private const string ActiveSessionKey = "JGuard_ActiveSessionId";
    
    private ApplicationDataContainer? _localSettings;
    private bool _initialized = false;
    private bool _isUnpackaged = false;
    private string? _fallbackFilePath;
    private List<Session> _cachedSessions = new();

    public SessionManager()
    {
    }

    /// <summary>
    /// Initialize the SessionManager with async retry for ApplicationData access.
    /// Falls back to local file storage for unpackaged apps.
    /// </summary>
    public async Task InitializeAsync()
    {
        if (_initialized) return;
        
        // Detect if we are running unpackaged
        try 
        {
            var folder = ApplicationData.Current.LocalFolder;
            _localSettings = ApplicationData.Current.LocalSettings;
            _isUnpackaged = false;
        }
        catch (InvalidOperationException)
        {
            _isUnpackaged = true;
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string jGuardPath = Path.Combine(appData, "JGuard");
            if (!Directory.Exists(jGuardPath)) Directory.CreateDirectory(jGuardPath);
            _fallbackFilePath = Path.Combine(jGuardPath, "sessions.json");
        }

        if (!_isUnpackaged)
        {
            _initialized = true;
            return;
        }

        // Handle unpackaged initialization
        if (File.Exists(_fallbackFilePath))
        {
            try
            {
                string json = await File.ReadAllTextAsync(_fallbackFilePath);
                _cachedSessions = JsonSerializer.Deserialize<List<Session>>(json) ?? new List<Session>();
            }
            catch { }
        }
        _initialized = true;
    }

    private ApplicationDataContainer LocalSettings
    {
        get
        {
            if (_isUnpackaged) throw new InvalidOperationException("Using fallback storage.");
            if (_localSettings == null)
            {
                throw new InvalidOperationException("SessionManager not initialized. Call InitializeAsync() first.");
            }
            return _localSettings;
        }
    }

    /// <summary>
    /// Save a session to local storage
    /// </summary>
    public void SaveSession(Session session)
    {
        var sessions = GetAllSessions();
        
        // Remove if exists, then add
        sessions = sessions.Where(s => s.SessionId != session.SessionId).ToList();
        sessions.Add(session);
        
        if (_isUnpackaged && _fallbackFilePath != null)
        {
            var json = JsonSerializer.Serialize(sessions);
            File.WriteAllText(_fallbackFilePath, json);
            _cachedSessions = sessions;
            return;
        }

        var jsonStr = JsonSerializer.Serialize(sessions);
        LocalSettings.Values[SessionsKey] = jsonStr;
    }

    /// <summary>
    /// Get all saved sessions from local storage
    /// </summary>
    public List<Session> GetAllSessions()
    {
        if (_isUnpackaged) return _cachedSessions;

        if (LocalSettings.Values.TryGetValue(SessionsKey, out var json))
        {
            try
            {
                var sessions = JsonSerializer.Deserialize<List<Session>>(json as string ?? "[]");
                return sessions ?? new List<Session>();
            }
            catch
            {
                return new List<Session>();
            }
        }
        return new List<Session>();
    }

    /// <summary>
    /// Get a specific session by ID
    /// </summary>
    public Session? GetSessionById(string sessionId)
    {
        return GetAllSessions().FirstOrDefault(s => s.SessionId == sessionId);
    }

    /// <summary>
    /// Delete a session from local storage
    /// </summary>
    public void DeleteSession(string sessionId)
    {
        var sessions = GetAllSessions();
        sessions = sessions.Where(s => s.SessionId != sessionId).ToList();
        
        if (_isUnpackaged && _fallbackFilePath != null)
        {
            if (sessions.Count == 0)
            {
                if (File.Exists(_fallbackFilePath)) File.Delete(_fallbackFilePath);
            }
            else
            {
                var json = JsonSerializer.Serialize(sessions);
                File.WriteAllText(_fallbackFilePath, json);
            }
            _cachedSessions = sessions;
            return;
        }

        if (sessions.Count == 0)
        {
            LocalSettings.Values.Remove(SessionsKey);
        }
        else
        {
            var json = JsonSerializer.Serialize(sessions);
            LocalSettings.Values[SessionsKey] = json;
        }
    }

    /// <summary>
    /// Set the active session ID
    /// </summary>
    public void SetActiveSession(string sessionId)
    {
        if (_isUnpackaged)
        {
            // We could save this to a separate file, but for now we'll just use it in memory 
            // since active session is usually set during the run.
            return;
        }
        LocalSettings.Values[ActiveSessionKey] = sessionId;
    }

    /// <summary>
    /// Get the active session ID
    /// </summary>
    public string? GetActiveSessionId()
    {
        if (_isUnpackaged) return null;

        if (LocalSettings.Values.TryGetValue(ActiveSessionKey, out var sessionId))
        {
            return sessionId as string;
        }
        return null;
    }

    /// <summary>
    /// Clear the active session
    /// </summary>
    public void ClearActiveSession()
    {
        LocalSettings.Values.Remove(ActiveSessionKey);
    }
}
