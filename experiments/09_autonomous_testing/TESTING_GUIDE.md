# Autonomous Game Testing - Complete Guide

**Automate your QA with AI-powered testing agents**

---

## 📖 Table of Contents

1. [Quick Start](#-quick-start)
2. [Agent Types](#-agent-types)
3. [Unity Integration](#-unity-integration)
4. [Running Tests](#-running-tests)
5. [Analyzing Results](#-analyzing-results)
6. [Best Practices](#-best-practices)
7. [Troubleshooting](#-troubleshooting)
8. [Advanced Usage](#-advanced-usage)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd experiments/09_autonomous_testing
pip install -r requirements.txt
```

### 2. Run Your First Test (Demo)

```python
# demo_test.py
from exploration_agent import ExplorationAgent

# Create agent
agent = ExplorationAgent(
    exploration_strategy='curiosity',
    verbose=True
)

# Set available actions
agent.set_available_actions(['UP', 'DOWN', 'LEFT', 'RIGHT'])

# Define dummy game functions
def get_state():
    return {'player_position': [0, 0], 'health': 100}

def take_action(action):
    print(f"Executing: {action}")

def is_terminal(state):
    return state['health'] <= 0

# Run test
results = agent.run(
    get_state_fn=get_state,
    take_action_fn=take_action,
    is_terminal_fn=is_terminal,
    episodes=10,
    max_steps=50
)

print(f"\n✓ Test completed!")
print(f"  Episodes: {results['summary']['episodes_completed']}")
print(f"  States visited: {results['summary']['unique_states_visited']}")
print(f"  Bugs found: {results['summary']['bugs_found']}")
```

### 3. View Results

```bash
python demo_test.py
```

---

## 🤖 Agent Types

### 1. Exploration Agent

**Purpose**: Maximum state coverage, find hidden areas

**When to use**:
- New level testing
- Finding unreachable areas
- Coverage analysis
- Discovering hidden content

**Example**:

```python
from exploration_agent import ExplorationAgent

agent = ExplorationAgent(
    exploration_strategy='curiosity',  # 'random', 'curiosity', 'epsilon_greedy'
    epsilon=0.3,
    novelty_threshold=0.8
)

results = agent.run(
    get_state_fn=game.get_state,
    take_action_fn=game.take_action,
    is_terminal_fn=game.is_terminal,
    episodes=1000
)

# Coverage report
coverage = agent.get_coverage_report()
print(f"States discovered: {coverage['total_states_discovered']}")
print(f"Coverage: {coverage['state_coverage']:.1f}%")
```

**Strategies**:
- **Random**: Completely random exploration
- **Curiosity**: Prioritizes unexplored areas
- **Epsilon-Greedy**: Balances exploration and exploitation

### 2. Bug Hunter Agent

**Purpose**: Find crashes, edge cases, and exploits

**When to use**:
- Pre-release testing
- Regression testing
- Finding crash bugs
- Stress testing

**Example**:

```python
from bug_hunter_agent import BugHunterAgent

agent = BugHunterAgent(
    aggression_level=0.9,  # 0.0-1.0
    spam_probability=0.5,
    boundary_testing=True,
    rapid_input=True
)

results = agent.run(
    get_state_fn=game.get_state,
    take_action_fn=game.take_action,
    is_terminal_fn=game.is_terminal,
    episodes=500
)

# Bug report
report = agent.get_bug_hunter_report()
print(f"Total bugs: {report['total_bugs_found']}")
print(f"Critical: {report['bug_breakdown']['critical']}")
print(f"High: {report['bug_breakdown']['high']}")
```

**Tactics**:
- Input spamming (same action repeatedly)
- Rapid action switching
- Boundary value testing
- Invalid action combinations
- Resource exhaustion

### 3. Progression Agent

**Purpose**: Verify levels are completable, measure difficulty

**When to use**:
- Level validation
- Difficulty balancing
- Softlock detection
- Optimal path finding

**Example**:

```python
from progression_agent import ProgressionAgent

agent = ProgressionAgent(
    goal_state={'level': 'Level_2'},  # Define goal
    max_attempts_per_episode=3,
    search_strategy='mixed'  # 'astar', 'greedy', 'mixed', 'random'
)

results = agent.run(
    get_state_fn=game.get_state,
    take_action_fn=game.take_action,
    is_terminal_fn=game.is_terminal,
    episodes=100
)

# Progression report
report = agent.get_progression_report()
print(f"Completion rate: {report['completion_rate']:.1%}")
print(f"Difficulty: {report['difficulty']}/10")
print(f"Avg attempts: {report['avg_attempts']:.2f}")

if report['diagnosis']:
    for msg in report['diagnosis']:
        print(msg)
```

**Search Strategies**:
- **A-star**: Goal-oriented with heuristics
- **Greedy**: Always choose best immediate action
- **Mixed**: 80% goal-oriented, 20% random exploration
- **Random**: Baseline comparison

---

## 🎮 Unity Integration

### Step 1: Add TestingBridge to Unity

1. Copy `4_integration/TestingBridge.cs` to your Unity project
2. Create empty GameObject: `GameObject` → `Create Empty`
3. Add component: `Add Component` → `Testing Bridge`
4. Configure:
   - **Port**: 5555 (default)
   - **Auto Start**: ✓ (checked)
   - **Player Object**: Drag your player GameObject here

### Step 2: Implement Game-Specific Methods

Edit `TestingBridge.cs` to match your game:

```csharp
private void UpdateGameState()
{
    // Update with your game's actual values
    currentState.player_position = PlayerController.Instance.transform.position;
    currentState.health = PlayerController.Instance.health;
    currentState.score = GameManager.Instance.score;
    currentState.level = SceneManager.GetActiveScene().name;
    currentState.enemies = FindObjectsOfType<Enemy>().Length;
}

private void ExecuteAction(string action)
{
    // Implement your game's actions
    switch (action)
    {
        case "JUMP":
            PlayerController.Instance.Jump();
            break;

        case "MOVE_LEFT":
            PlayerController.Instance.MoveDirection(Vector2.left);
            break;

        case "MOVE_RIGHT":
            PlayerController.Instance.MoveDirection(Vector2.right);
            break;

        case "ATTACK":
            PlayerController.Instance.Attack();
            break;
    }
}

private void CheckTerminalState()
{
    // Define your end conditions
    isTerminalState =
        PlayerController.Instance.health <= 0 ||  // Player died
        GameManager.Instance.levelComplete;        // Level complete
}
```

### Step 3: Run Tests from Python

```python
from unity_bridge import UnityBridge, run_unity_test
from exploration_agent import ExplorationAgent

# Run test
results = run_unity_test(
    agent_class=ExplorationAgent,
    agent_kwargs={
        'exploration_strategy': 'curiosity',
        'verbose': True
    },
    episodes=100,
    unity_port=5555,
    time_scale=10.0,  # 10x speed
    output_dir='results/my_game_test'
)
```

### Step 4: Review Results

Results are saved to `results/my_game_test/unity_test_results.json`

---

## 🏃 Running Tests

### Test Workflow

```
┌─────────────┐
│ Setup Agent │
└──────┬──────┘
       │
┌──────▼──────────┐
│ Connect to Game │
└──────┬──────────┘
       │
┌──────▼──────────┐
│ Run Episodes    │ ◄─┐
│ - Get state     │   │
│ - Select action │   │ Repeat
│ - Execute       │   │ N episodes
│ - Detect bugs   │   │
└──────┬──────────┘   │
       │              │
       └──────────────┘
       │
┌──────▼──────────┐
│ Generate Report │
└─────────────────┘
```

### Example: Full Test Suite

```python
# full_test_suite.py
from exploration_agent import ExplorationAgent
from bug_hunter_agent import BugHunterAgent
from progression_agent import ProgressionAgent
from unity_bridge import UnityBridge
from bug_reporter import BugReporter
from coverage_analyzer import CoverageAnalyzer

def run_full_test_suite(game_build_path=None):
    """Run complete test suite on Unity game."""

    # 1. Exploration Test
    print("=" * 60)
    print("PHASE 1: Exploration Test")
    print("=" * 60)

    exploration_results = run_unity_test(
        agent_class=ExplorationAgent,
        episodes=500,
        time_scale=10.0,
        output_dir='results/exploration'
    )

    # 2. Bug Hunting Test
    print("\n" + "=" * 60)
    print("PHASE 2: Bug Hunting Test")
    print("=" * 60)

    bug_hunter_results = run_unity_test(
        agent_class=BugHunterAgent,
        episodes=300,
        time_scale=5.0,  # Slower for bug detection
        output_dir='results/bug_hunting'
    )

    # 3. Progression Test
    print("\n" + "=" * 60)
    print("PHASE 3: Progression Test")
    print("=" * 60)

    progression_results = run_unity_test(
        agent_class=ProgressionAgent,
        agent_kwargs={
            'goal_state': {'level': 'Level_Complete'}
        },
        episodes=100,
        time_scale=10.0,
        output_dir='results/progression'
    )

    # 4. Generate Reports
    print("\n" + "=" * 60)
    print("PHASE 4: Report Generation")
    print("=" * 60)

    # Bug report
    reporter = BugReporter(output_dir='reports')
    reporter.output_format = 'html'
    reporter.generate_report(bug_hunter_results, 'bug_report.html')

    # Coverage report
    analyzer = CoverageAnalyzer()
    coverage = analyzer.analyze(exploration_results)
    analyzer.generate_report(coverage, output_file='coverage_report.html')

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print(f"\nExploration:")
    print(f"  States: {exploration_results['summary']['unique_states_visited']}")
    print(f"  Coverage: {exploration_results['summary']['state_coverage']:.1f}%")
    print(f"\nBug Hunting:")
    print(f"  Bugs found: {bug_hunter_results['summary']['bugs_found']}")
    print(f"\nProgression:")
    prog_report = progression_results.get('progression_report', {})
    print(f"  Completion rate: {prog_report.get('completion_rate', 0):.1%}")
    print(f"  Difficulty: {prog_report.get('difficulty', 0):.1f}/10")

    print(f"\n✓ Reports saved to 'reports/' directory")

if __name__ == "__main__":
    run_full_test_suite()
```

---

## 📊 Analyzing Results

### Bug Report

Generated as HTML/Markdown/JSON:

```
reports/
├── bug_report.html          # HTML report (open in browser)
├── bug_report.md            # Markdown report
└── bug_report.json          # JSON data
```

**Bug Report Contents**:
- Summary statistics
- Bugs by severity (Critical/High/Medium/Low)
- Steps to reproduce
- Game state snapshots
- Timestamps

### Coverage Report

```
reports/
└── coverage_report.html     # Interactive coverage visualization
```

**Coverage Report Contents**:
- Total states visited
- State visitation distribution
- Unreached areas estimate
- Hotspots (frequently visited)
- Dead zones (visited once)

### JSON Results

All agents save JSON results with full details:

```json
{
  "agent_name": "ExplorationAgent",
  "timestamp": "2025-01-15 14:30:00",
  "summary": {
    "episodes_completed": 1000,
    "total_steps": 50000,
    "unique_states_visited": 2500,
    "bugs_found": 12,
    "state_coverage": 78.5
  },
  "bugs": [
    {
      "bug_id": "BUG-1234567890",
      "severity": "critical",
      "title": "Player falls through floor",
      "description": "...",
      "game_state": {...},
      "steps_to_reproduce": [...]
    }
  ]
}
```

---

## 💡 Best Practices

### 1. Start Small

```python
# First run: Quick smoke test (10 episodes)
results = agent.run(episodes=10, max_steps=100)

# If successful: Medium test (100 episodes)
results = agent.run(episodes=100, max_steps=500)

# Production: Full test (1000+ episodes)
results = agent.run(episodes=1000, max_steps=1000)
```

### 2. Use Time Scaling

```python
# Speed up testing 10x
bridge.set_time_scale(10.0)

# Slow down for bug detection
bridge.set_time_scale(2.0)

# Reset to normal
bridge.set_time_scale(1.0)
```

### 3. Test Incrementally

```
Day 1: Exploration agent → Find coverage gaps
Day 2: Bug hunter → Find crashes
Day 3: Progression → Verify completability
Day 4: Fix bugs
Day 5: Re-test with all agents
```

### 4. Integrate with CI/CD

```yaml
# .github/workflows/game-testing.yml
name: Automated Game Testing

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Run Unity tests
        run: |
          python full_test_suite.py

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: reports/
```

### 5. Filter Bugs by Severity

```python
# Only report critical/high bugs
reporter = BugReporter()
reporter.create_jira_issues(
    agent_results,
    severity_filter=['critical', 'high']
)
```

---

## 🔧 Troubleshooting

### Cannot Connect to Unity

**Problem**: `✗ Failed to connect to Unity`

**Solutions**:
1. Check Unity is running in Play mode
2. Verify `TestingBridge` component is attached
3. Check port number matches (default 5555)
4. Disable firewall for localhost
5. Check console for Unity errors

```python
# Test connection
from unity_bridge import UnityBridge

bridge = UnityBridge(port=5555)
if bridge.connect():
    print("✓ Connected!")
    state = bridge.get_state()
    print(f"State: {state}")
else:
    print("✗ Connection failed")
```

### Agent Finds No Bugs

**Problem**: Bug hunter reports 0 bugs (but you know there are bugs)

**Solutions**:
1. Increase aggression level:
```python
agent = BugHunterAgent(aggression_level=1.0)  # Max aggression
```

2. Increase episodes:
```python
results = agent.run(episodes=1000)  # More attempts
```

3. Check bug detection methods are implemented in `TestingBridge.cs`

### Low Coverage

**Problem**: Coverage report shows <50% coverage

**Solutions**:
1. Use curiosity-driven strategy:
```python
agent = ExplorationAgent(exploration_strategy='curiosity')
```

2. Increase episodes:
```python
results = agent.run(episodes=2000)
```

3. Adjust novelty threshold:
```python
agent = ExplorationAgent(novelty_threshold=0.9)
```

### Slow Testing

**Problem**: Tests take too long

**Solutions**:
1. Increase time scale:
```python
bridge.set_time_scale(20.0)  # 20x speed
```

2. Reduce max steps:
```python
agent.run(max_steps=200)  # Shorter episodes
```

3. Run headless Unity build (no rendering)

---

## 🎓 Advanced Usage

### Custom Bug Detection

Extend `BaseTestingAgent` to add custom bug detection:

```python
from base_agent import BaseTestingAgent, Bug, create_bug

class MyCustomAgent(BaseTestingAgent):
    def select_action(self, state):
        # Your action selection logic
        return 'JUMP'

    def detect_bugs(self, state, next_state):
        bugs = []

        # Custom bug: player moving too fast
        prev_pos = state.get('player_position')
        next_pos = next_state.get('player_position')

        if prev_pos and next_pos:
            distance = ((next_pos[0] - prev_pos[0])**2 +
                       (next_pos[1] - prev_pos[1])**2)**0.5

            if distance > 100:  # Threshold
                bugs.append(create_bug(
                    severity='high',
                    title='Abnormal movement speed',
                    description=f'Player moved {distance:.2f} units in one step',
                    state=state
                ))

        return bugs
```

### Multi-Level Testing

Test multiple levels automatically:

```python
from unity_bridge import UnityBridge
from exploration_agent import ExplorationAgent

bridge = UnityBridge()
bridge.connect()

levels = ['Level_1', 'Level_2', 'Level_3']

for level_name in levels:
    print(f"\nTesting {level_name}...")

    # Load level
    bridge.load_scene(level_name)
    time.sleep(2)  # Wait for load

    # Run test
    agent = ExplorationAgent()
    agent.set_available_actions(['JUMP', 'MOVE_LEFT', 'MOVE_RIGHT'])

    results = agent.run(
        get_state_fn=bridge.get_state,
        take_action_fn=bridge.take_action,
        is_terminal_fn=bridge.is_terminal,
        episodes=100
    )

    agent.save_results(f"results/{level_name}_results.json")

bridge.disconnect()
```

### Slack/JIRA Integration

Auto-report bugs to your team:

```python
from bug_reporter import BugReporter

reporter = BugReporter(
    jira_url='https://your-company.atlassian.net',
    jira_project='GAME',
    jira_username='bot@company.com',
    jira_api_token='YOUR_API_TOKEN',
    slack_webhook='https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
)

# Run tests
results = agent.run(...)

# Create JIRA tickets for critical bugs
reporter.create_jira_issues(results, severity_filter=['critical'])

# Send Slack notification
reporter.send_slack_notification(results)
```

---

## 📚 Further Reading

- [Base Agent API](1_agents/base_agent.py)
- [Unity Integration Examples](4_integration/)
- [Bug Report Formats](3_reporting/bug_reporter.py)
- [Main README](README.md)

---

**Happy Testing! 🎮🤖🚀**
