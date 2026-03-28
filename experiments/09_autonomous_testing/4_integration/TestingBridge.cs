/*
 * TestingBridge.cs - Unity側
 *
 * Pythonテストエージェントと通信するためのUnity TCPサーバー.
 *
 * 使い方:
 *   1. このスクリプトを Unity プロジェクトに配置
 *   2. 空のGameObjectにアタッチ
 *   3. Playモードで自動的にサーバー起動
 *   4. Python側から接続してテスト実行
 *
 * 設定:
 *   - Port: TCPポート番号（デフォルト 5555）
 *   - Auto Start: Playモードで自動起動
 */

using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;

[System.Serializable]
public class GameStateData
{
    public Vector3 player_position;
    public int health;
    public int score;
    public string level;
    public int enemies;
}

[System.Serializable]
public class CommandMessage
{
    public string command;
    public Dictionary<string, object> data;
}

[System.Serializable]
public class ResponseMessage
{
    public string status;  // "OK", "ERROR", "TIMEOUT"
    public string message;
    public Dictionary<string, object> data;
}

public class TestingBridge : MonoBehaviour
{
    [Header("Server Settings")]
    [SerializeField] private int port = 5555;
    [SerializeField] private bool autoStart = true;

    [Header("Game References")]
    [SerializeField] private GameObject playerObject;

    [Header("Status")]
    [SerializeField] private bool isRunning = false;
    [SerializeField] private string connectionStatus = "Not started";

    private TcpListener tcpListener;
    private TcpClient connectedClient;
    private NetworkStream clientStream;
    private Thread listenerThread;
    private bool shouldStop = false;

    // ゲーム状態
    private GameStateData currentState = new GameStateData();
    private bool isTerminalState = false;

    void Start()
    {
        if (autoStart)
        {
            StartServer();
        }
    }

    void OnDestroy()
    {
        StopServer();
    }

    public void StartServer()
    {
        if (isRunning)
        {
            Debug.LogWarning("[TestingBridge] Server already running");
            return;
        }

        shouldStop = false;
        listenerThread = new Thread(ListenForClients);
        listenerThread.IsBackground = true;
        listenerThread.Start();

        isRunning = true;
        Debug.Log($"[TestingBridge] Server started on port {port}");
    }

    public void StopServer()
    {
        shouldStop = true;

        if (connectedClient != null)
        {
            connectedClient.Close();
            connectedClient = null;
        }

        if (tcpListener != null)
        {
            tcpListener.Stop();
            tcpListener = null;
        }

        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Join(1000);
        }

        isRunning = false;
        connectionStatus = "Stopped";
        Debug.Log("[TestingBridge] Server stopped");
    }

    private void ListenForClients()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, port);
            tcpListener.Start();

            Debug.Log($"[TestingBridge] Listening on port {port}...");
            connectionStatus = $"Listening on port {port}";

            while (!shouldStop)
            {
                if (tcpListener.Pending())
                {
                    connectedClient = tcpListener.AcceptTcpClient();
                    clientStream = connectedClient.GetStream();

                    connectionStatus = "Client connected";
                    Debug.Log("[TestingBridge] Client connected");

                    HandleClient();

                    connectionStatus = "Client disconnected";
                    Debug.Log("[TestingBridge] Client disconnected");
                }
                else
                {
                    Thread.Sleep(100);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[TestingBridge] Server error: {e.Message}");
            connectionStatus = $"Error: {e.Message}";
        }
    }

    private void HandleClient()
    {
        byte[] buffer = new byte[4096];

        try
        {
            while (!shouldStop && connectedClient.Connected)
            {
                // メッセージ長を受信（4バイト）
                byte[] lengthBytes = new byte[4];
                int bytesRead = clientStream.Read(lengthBytes, 0, 4);

                if (bytesRead == 0)
                    break;

                if (BitConverter.IsLittleEndian)
                    Array.Reverse(lengthBytes);

                int messageLength = BitConverter.ToInt32(lengthBytes, 0);

                // メッセージ本体を受信
                byte[] messageBytes = new byte[messageLength];
                int totalRead = 0;

                while (totalRead < messageLength)
                {
                    int read = clientStream.Read(messageBytes, totalRead, messageLength - totalRead);
                    if (read == 0)
                        break;
                    totalRead += read;
                }

                string messageJson = Encoding.UTF8.GetString(messageBytes);

                // コマンド処理
                string responseJson = ProcessCommand(messageJson);

                // レスポンス送信
                SendResponse(responseJson);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[TestingBridge] Client handling error: {e.Message}");
        }
        finally
        {
            if (clientStream != null)
                clientStream.Close();
            if (connectedClient != null)
                connectedClient.Close();
        }
    }

    private string ProcessCommand(string messageJson)
    {
        try
        {
            var message = JsonUtility.FromJson<CommandMessage>(messageJson);
            string command = message.command;

            Debug.Log($"[TestingBridge] Received command: {command}");

            switch (command)
            {
                case "PING":
                    return CreateResponse("OK", "Pong");

                case "GET_STATE":
                    UpdateGameState();
                    var stateJson = JsonUtility.ToJson(currentState);
                    return CreateResponse("OK", "State retrieved", new Dictionary<string, object>
                    {
                        { "state", JsonUtility.FromJson<Dictionary<string, object>>(stateJson) }
                    });

                case "EXECUTE_ACTION":
                    if (message.data != null && message.data.ContainsKey("action"))
                    {
                        string action = message.data["action"].ToString();
                        ExecuteAction(action);
                        return CreateResponse("OK", $"Action '{action}' executed");
                    }
                    return CreateResponse("ERROR", "No action specified");

                case "IS_TERMINAL":
                    CheckTerminalState();
                    return CreateResponse("OK", "Terminal state checked", new Dictionary<string, object>
                    {
                        { "terminal", isTerminalState }
                    });

                case "RESET":
                    ResetGame();
                    return CreateResponse("OK", "Game reset");

                case "SCREENSHOT":
                    if (message.data != null && message.data.ContainsKey("path"))
                    {
                        string path = message.data["path"].ToString();
                        CaptureScreenshot(path);
                        return CreateResponse("OK", $"Screenshot saved to {path}");
                    }
                    return CreateResponse("ERROR", "No path specified");

                case "SET_TIME_SCALE":
                    if (message.data != null && message.data.ContainsKey("scale"))
                    {
                        float scale = Convert.ToSingle(message.data["scale"]);
                        Time.timeScale = scale;
                        return CreateResponse("OK", $"Time scale set to {scale}");
                    }
                    return CreateResponse("ERROR", "No scale specified");

                case "LOAD_SCENE":
                    if (message.data != null && message.data.ContainsKey("scene"))
                    {
                        string sceneName = message.data["scene"].ToString();
                        SceneManager.LoadScene(sceneName);
                        return CreateResponse("OK", $"Loading scene {sceneName}");
                    }
                    return CreateResponse("ERROR", "No scene specified");

                case "DISCONNECT":
                    return CreateResponse("OK", "Disconnecting");

                default:
                    return CreateResponse("ERROR", $"Unknown command: {command}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[TestingBridge] Command processing error: {e.Message}");
            return CreateResponse("ERROR", e.Message);
        }
    }

    private void SendResponse(string responseJson)
    {
        try
        {
            byte[] responseBytes = Encoding.UTF8.GetBytes(responseJson);
            byte[] lengthBytes = BitConverter.GetBytes(responseBytes.Length);

            if (BitConverter.IsLittleEndian)
                Array.Reverse(lengthBytes);

            clientStream.Write(lengthBytes, 0, 4);
            clientStream.Write(responseBytes, 0, responseBytes.Length);
            clientStream.Flush();
        }
        catch (Exception e)
        {
            Debug.LogError($"[TestingBridge] Response send error: {e.Message}");
        }
    }

    private string CreateResponse(string status, string message, Dictionary<string, object> data = null)
    {
        var response = new ResponseMessage
        {
            status = status,
            message = message,
            data = data ?? new Dictionary<string, object>()
        };

        return JsonUtility.ToJson(response);
    }

    private void UpdateGameState()
    {
        // プレイヤー位置
        if (playerObject != null)
        {
            currentState.player_position = playerObject.transform.position;
        }

        // TODO: 実際のゲームロジックに合わせて実装
        // これはサンプル実装
        currentState.health = 100;  // PlayerController.Instance.health など
        currentState.score = 0;     // GameManager.Instance.score など
        currentState.level = SceneManager.GetActiveScene().name;
        currentState.enemies = FindObjectsOfType<EnemyController>().Length;  // 敵の数
    }

    private void ExecuteAction(string action)
    {
        // TODO: 実際のゲームロジックに合わせて実装
        // これはサンプル実装

        switch (action)
        {
            case "JUMP":
                // PlayerController.Instance.Jump();
                Debug.Log("[TestingBridge] Action: JUMP");
                break;

            case "MOVE_LEFT":
                // PlayerController.Instance.Move(Vector2.left);
                Debug.Log("[TestingBridge] Action: MOVE_LEFT");
                break;

            case "MOVE_RIGHT":
                // PlayerController.Instance.Move(Vector2.right);
                Debug.Log("[TestingBridge] Action: MOVE_RIGHT");
                break;

            case "ATTACK":
                // PlayerController.Instance.Attack();
                Debug.Log("[TestingBridge] Action: ATTACK");
                break;

            default:
                Debug.LogWarning($"[TestingBridge] Unknown action: {action}");
                break;
        }
    }

    private void CheckTerminalState()
    {
        // TODO: 実際のゲームロジックに合わせて実装
        // 終了条件をチェック

        // 例: 健康値が0以下
        isTerminalState = currentState.health <= 0;

        // 例: ゴール到達
        // isTerminalState = playerObject.transform.position == goalPosition;
    }

    private void ResetGame()
    {
        // TODO: 実際のゲームロジックに合わせて実装
        // ゲームをリセット

        // 例: シーンリロード
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);

        Debug.Log("[TestingBridge] Game reset");
    }

    private void CaptureScreenshot(string path)
    {
        ScreenCapture.CaptureScreenshot(path);
        Debug.Log($"[TestingBridge] Screenshot captured: {path}");
    }

    // Unity Editor用のGUI
    void OnGUI()
    {
        if (!isRunning)
            return;

        GUILayout.BeginArea(new Rect(10, 10, 300, 100));
        GUILayout.Box($"Testing Bridge\nStatus: {connectionStatus}\nPort: {port}");
        GUILayout.EndArea();
    }
}

// ダミークラス（実際のプロジェクトでは実装されているはず）
public class EnemyController : MonoBehaviour
{
    // Placeholder
}
