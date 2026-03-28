# Validate All Command

Run comprehensive validation on code, data, and experiments.

## Usage

When the user types `/validate-all`:

1. **Code Quality Checks**:
   - Python syntax validation
   - PEP 8 style compliance (with flake8 or black)
   - Type hint coverage
   - Docstring coverage
   - TODO/FIXME comments

2. **Dependency Checks**:
   - Verify all requirements.txt packages are installed
   - Check for version conflicts
   - Identify unused imports
   - Security vulnerabilities (with safety)

3. **Data Validation**:
   - Check data/ directory structure
   - Verify dataset integrity
   - Check for corrupted files
   - Validate data splits (train/val/test)

4. **Model Validation**:
   - Check saved models exist and are loadable
   - Verify model architectures match saved checkpoints
   - Check for NaN/Inf weights

5. **Experiment Validation**:
   - Each experiment has required files (train.py, model.py, README.md)
   - Training scripts are runnable
   - No broken imports
   - Consistent naming conventions

6. **Git Repository Health**:
   - No large files committed (>100MB)
   - No sensitive data (API keys, passwords)
   - .gitignore properly configured
   - Clean working directory

## Validation Categories

### Critical Issues ⛔
Must fix before continuing:
- Syntax errors
- Missing dependencies
- Corrupted data files
- Broken imports

### High Priority ⚠️
Should fix soon:
- Style violations
- Missing docstrings
- Unused imports
- Security vulnerabilities

### Medium Priority 📋
Nice to fix:
- Type hint coverage
- TODO comments
- Inconsistent naming

### Low Priority 💡
Optional improvements:
- Code comments
- Additional documentation
- Optimization opportunities

## Output Format

```
=== AiModelPlayground Validation Report ===
Date: 2026-03-25 12:30:00

[1/6] Code Quality Checks...
✅ Python syntax: OK (10 files checked)
⚠️  PEP 8 compliance: 5 style issues found
✅ Type hints: 80% coverage
⚠️  Docstrings: 60% coverage (needs improvement)
📋 TODO comments: 3 found

[2/6] Dependency Checks...
✅ All requirements installed
✅ No version conflicts
⚠️  Unused imports: 2 found
✅ No security vulnerabilities

[3/6] Data Validation...
✅ data/ directory exists
✅ MNIST dataset: OK (70000 samples)
✅ No corrupted files

[4/6] Model Validation...
✅ models/mnist_simple.pth: loadable
⚠️  models/old_model.pth: architecture mismatch

[5/6] Experiment Validation...
✅ 01_mnist_basics: Complete (train.py, model.py, README.md)
⚠️  02_sentiment_analysis: Missing train.py
⚠️  03_transfer_learning: Empty directory

[6/6] Git Repository Health...
✅ No large files committed
✅ No sensitive data found
✅ .gitignore properly configured
⚠️  Untracked files: 3 (consider adding or ignoring)

=====================================
Summary:
  Critical: 0 ⛔
  High:     2 ⚠️
  Medium:   4 📋
  Low:      3 💡

Overall Status: PASS (with warnings)

Details:
[HIGH] experiments/02_sentiment_analysis/train.py missing
  → Run: /experiment-create sentiment-analysis

[HIGH] models/old_model.pth architecture mismatch
  → Remove old checkpoint or update architecture

[MEDIUM] 5 PEP 8 style violations
  → Run: black . (to auto-format)

[MEDIUM] Docstring coverage 60%
  → Add docstrings to: model.py, utils.py

Full report saved: validation_report.txt
```

## Auto-fix Options

Offer to automatically fix common issues:
- Format code with Black
- Remove unused imports
- Add basic docstrings
- Update .gitignore

```
Would you like me to:
1. Auto-format code with Black? (y/n)
2. Remove unused imports? (y/n)
3. Generate basic docstrings? (y/n)
```

## Integration

Run this command:
- Before committing changes
- Before starting new experiments
- Weekly as maintenance
- Before project reviews
