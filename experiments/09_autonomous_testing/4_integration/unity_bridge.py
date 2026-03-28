"""
Unity Bridge - Python側

UnityゲームとPythonテストエージェントを接続するブリッジ.

使い方:
    from unity_bridge import UnityBridge
    from exploration_agent import ExplorationAgent

    # ブリッジ作成
    bridge = UnityBridge(port=5555)

    # Unityに接続
    bridge.connect()

    # エージェント作成
    agent = ExplorationAgent()
    agent.set_available_actions(['JUMP', 'MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK'])

    # テスト実行
    results = agent.run(
        get_state_fn=bridge.get_state,
        take_action_fn=bridge.take_action,
        is_terminal_fn=bridge.is_terminal,
        episodes=100
    )

    bridge.disconnect()

Unity側の実装:
    TestingBridge.cs を Unity プロジェクトに配置し、
    TCPサーバーを起動してください。
"""

import socket
import json
import time
import struct
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path


class UnityBridge:
    """
    Unity ゲームとの通信ブリッジ（Python側）.

    プロトコル:
        - TCP/IPソケット通信
        - JSON形式でメッセージ交換
        - メッセージフォーマット: [4バイト長さ][JSONデータ]
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5555,
        timeout: float = 10.0,
        verbose: bool = True
    ):
        """
        Args:
            host: Unityサーバーのホスト
            port: ポート番号
            timeout: 接続タイムアウト（秒）
            verbose: 詳細ログ
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.verbose = verbose

        self.socket: Optional[socket.socket] = None
        self.connected = False

    def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """
        Unityサーバーに接続.

        Args:
            max_retries: 最大再試行回数
            retry_delay: 再試行間隔（秒）

        Returns:
            接続成功したかどうか
        """
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.host, self.port))

                self.connected = True

                if self.verbose:
                    print(f"✓ Connected to Unity at {self.host}:{self.port}")

                # 接続確認
                response = self.send_command('PING')
                if response.get('status') == 'OK':
                    return True

            except (ConnectionRefusedError, socket.timeout) as e:
                if self.verbose:
                    print(f"✗ Connection attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"✗ Failed to connect after {max_retries} attempts")
                    return False

        return False

    def disconnect(self):
        """接続を切断."""
        if self.socket:
            try:
                self.send_command('DISCONNECT')
            except:
                pass

            self.socket.close()
            self.socket = None
            self.connected = False

            if self.verbose:
                print("✓ Disconnected from Unity")

    def send_command(self, command: str, data: Optional[Dict] = None) -> Dict:
        """
        Unityにコマンドを送信.

        Args:
            command: コマンド名
            data: コマンドデータ

        Returns:
            Unityからのレスポンス
        """
        if not self.connected or not self.socket:
            raise RuntimeError("Not connected to Unity")

        message = {
            'command': command,
            'data': data or {}
        }

        # JSONエンコード
        message_json = json.dumps(message).encode('utf-8')

        # メッセージ長（4バイト）+ メッセージ本体
        message_length = struct.pack('!I', len(message_json))

        try:
            # 送信
            self.socket.sendall(message_length + message_json)

            # レスポンス受信
            response_length_data = self._recv_exact(4)
            response_length = struct.unpack('!I', response_length_data)[0]

            response_json = self._recv_exact(response_length)
            response = json.loads(response_json.decode('utf-8'))

            return response

        except socket.timeout:
            print(f"✗ Command timed out: {command}")
            return {'status': 'TIMEOUT'}
        except Exception as e:
            print(f"✗ Command failed: {e}")
            return {'status': 'ERROR', 'message': str(e)}

    def _recv_exact(self, n: int) -> bytes:
        """
        正確にnバイト受信.

        Args:
            n: 受信バイト数

        Returns:
            受信データ
        """
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by Unity")
            data += chunk
        return data

    def get_state(self) -> Dict:
        """
        現在のゲーム状態を取得.

        Returns:
            ゲーム状態の辞書
        """
        response = self.send_command('GET_STATE')

        if response.get('status') == 'OK':
            return response.get('state', {})
        else:
            print(f"✗ Failed to get state: {response.get('message')}")
            return {}

    def take_action(self, action: Any):
        """
        Unityでアクションを実行.

        Args:
            action: 実行するアクション（文字列または数値）
        """
        response = self.send_command('EXECUTE_ACTION', {'action': str(action)})

        if response.get('status') != 'OK':
            print(f"✗ Failed to execute action '{action}': {response.get('message')}")

    def is_terminal(self, state: Dict) -> bool:
        """
        終了状態かどうかを判定.

        Args:
            state: ゲーム状態

        Returns:
            終了状態ならTrue
        """
        # Unityに問い合わせ
        response = self.send_command('IS_TERMINAL')

        if response.get('status') == 'OK':
            return response.get('terminal', False)
        else:
            # デフォルト: 健康値が0以下
            return state.get('health', 100) <= 0

    def reset_game(self):
        """ゲームをリセット."""
        response = self.send_command('RESET')

        if response.get('status') == 'OK':
            if self.verbose:
                print("✓ Game reset")
        else:
            print(f"✗ Failed to reset game: {response.get('message')}")

    def capture_screenshot(self, filepath: str = 'screenshot.png') -> bool:
        """
        スクリーンショットを撮影.

        Args:
            filepath: 保存先パス

        Returns:
            成功したかどうか
        """
        response = self.send_command('SCREENSHOT', {'path': filepath})

        if response.get('status') == 'OK':
            if self.verbose:
                print(f"✓ Screenshot saved: {filepath}")
            return True
        else:
            print(f"✗ Failed to capture screenshot: {response.get('message')}")
            return False

    def set_time_scale(self, scale: float):
        """
        ゲームのタイムスケールを設定（高速化/スローモーション）.

        Args:
            scale: タイムスケール（1.0 = 通常速度）
        """
        response = self.send_command('SET_TIME_SCALE', {'scale': scale})

        if response.get('status') == 'OK':
            if self.verbose:
                print(f"✓ Time scale set to {scale}x")
        else:
            print(f"✗ Failed to set time scale: {response.get('message')}")

    def load_scene(self, scene_name: str):
        """
        シーンをロード.

        Args:
            scene_name: シーン名
        """
        response = self.send_command('LOAD_SCENE', {'scene': scene_name})

        if response.get('status') == 'OK':
            if self.verbose:
                print(f"✓ Scene loaded: {scene_name}")
        else:
            print(f"✗ Failed to load scene: {response.get('message')}")


# ヘルパー関数

def run_unity_test(
    agent_class,
    agent_kwargs: Dict = {},
    episodes: int = 100,
    max_steps: int = 1000,
    unity_host: str = 'localhost',
    unity_port: int = 5555,
    time_scale: float = 10.0,
    output_dir: str = 'results/unity_test'
) -> Dict:
    """
    Unityゲームでテストエージェントを実行.

    Args:
        agent_class: テストエージェントクラス
        agent_kwargs: エージェントの引数
        episodes: エピソード数
        max_steps: 最大ステップ数
        unity_host: Unityホスト
        unity_port: ポート
        time_scale: タイムスケール（テスト高速化）
        output_dir: 結果出力ディレクトリ

    Returns:
        テスト結果
    """
    # ブリッジ作成
    bridge = UnityBridge(host=unity_host, port=unity_port)

    # 接続
    if not bridge.connect():
        print("✗ Failed to connect to Unity")
        return {}

    try:
        # タイムスケール設定（テスト高速化）
        bridge.set_time_scale(time_scale)

        # エージェント作成
        agent = agent_class(**agent_kwargs)

        # テスト実行
        results = agent.run(
            get_state_fn=bridge.get_state,
            take_action_fn=bridge.take_action,
            is_terminal_fn=bridge.is_terminal,
            episodes=episodes,
            max_steps=max_steps
        )

        # 結果保存
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        agent.save_results(f"{output_dir}/unity_test_results.json")

        return results

    finally:
        # クリーンアップ
        bridge.set_time_scale(1.0)  # 通常速度に戻す
        bridge.disconnect()


if __name__ == "__main__":
    # デモ: Unity接続テスト
    print("Unity Bridge Demo")
    print("=" * 60)
    print("\n⚠️ This demo requires Unity to be running with TestingBridge.cs")
    print("   Start Unity with the testing server before running this demo.\n")

    # ブリッジ作成
    bridge = UnityBridge(port=5555)

    # 接続試行
    print("Connecting to Unity...")
    if bridge.connect(max_retries=3, retry_delay=1.0):
        print("\n✓ Connected!")

        # 状態取得テスト
        print("\nTesting state retrieval...")
        state = bridge.get_state()
        print(f"Current state: {state}")

        # アクション実行テスト
        print("\nTesting action execution...")
        bridge.take_action('JUMP')
        time.sleep(0.1)

        new_state = bridge.get_state()
        print(f"New state: {new_state}")

        # 終了判定テスト
        print("\nTesting terminal check...")
        is_done = bridge.is_terminal(new_state)
        print(f"Is terminal: {is_done}")

        # 切断
        bridge.disconnect()
        print("\n✓ Demo completed!")

    else:
        print("\n✗ Could not connect to Unity")
        print("   Make sure Unity is running with TestingBridge.cs on port 5555")
