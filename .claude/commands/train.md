# Train Model Command

Run training for a specific experiment with monitoring and error handling.

## Usage

When the user types `/train [experiment-name]`:

1. **Locate training script**:
   - Find `experiments/0X_experiment-name/train.py`
   - Verify file exists

2. **Pre-training checks**:
   - Check if GPU is available
   - Verify data directory exists
   - Check disk space for model saving
   - Verify all dependencies are installed

3. **Run training**:
   ```bash
   cd experiments/0X_experiment-name
   python train.py 2>&1 | tee training.log
   ```

4. **Monitor progress**:
   - Display training output in real-time
   - Watch for common errors:
     - CUDA out of memory
     - Data loading errors
     - NaN losses
     - Slow training (suggest GPU usage)

5. **Post-training**:
   - Show final metrics
   - Display location of saved model
   - Suggest visualization command
   - Offer to run evaluation

## Common Issues and Solutions

### CUDA Out of Memory
```
Error: RuntimeError: CUDA out of memory
Solution: Reduce batch size in train.py
```

### Slow Training on CPU
```
Warning: Training on CPU. GPU would be 10-100x faster.
Solution: Install CUDA toolkit or use Google Colab
```

### NaN Loss
```
Error: Loss became NaN
Possible causes:
- Learning rate too high (try 0.0001 instead of 0.001)
- Gradient explosion (add gradient clipping)
- Data normalization issue
```

### Data Not Found
```
Error: Dataset not found
Solution: Data will be auto-downloaded on first run
```

## Output Format

```
=== Training [Experiment Name] ===

Pre-flight checks:
✅ GPU available: NVIDIA GeForce RTX 3080
✅ Data directory: data/ (exists)
✅ Disk space: 50GB free
✅ Dependencies: OK

Starting training...

Epoch 1/10
Training: 100%|████████| 938/938 [00:15<00:00, loss=0.234, acc=93.1%]
Evaluating: 100%|████████| 157/157 [00:02<00:00]
Train Loss: 0.2518, Train Acc: 92.58%
Test Loss: 0.1234, Test Acc: 96.32%

...

Epoch 10/10
Training: 100%|████████| 938/938 [00:14<00:00, loss=0.045, acc=98.7%]
Evaluating: 100%|████████| 157/157 [00:02<00:00]
Train Loss: 0.0421, Train Acc: 98.76%
Test Loss: 0.0312, Test Acc: 97.89%

✅ Training complete!

Model saved: models/experiment-name.pth
Training log: experiments/0X_experiment-name/training.log

Next steps:
- Run: /evaluate experiment-name
- Run: /visualize experiment-name
```

## Additional Options

User can specify:
- `--epochs N` - Override epoch count
- `--batch-size N` - Override batch size
- `--lr FLOAT` - Override learning rate
- `--device cpu/cuda` - Force device

Example:
```
/train mnist --epochs 20 --batch-size 128
```
