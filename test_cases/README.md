# Test cases

The nine inputs are held-out SpanishBCBL sentence windows spanning easier, typical, and harder model predictions. `answers.csv` contains the recorded typed text and intended prompt. `results.json` contains fresh local RTX 4060 predictions.

Run all examples:

```powershell
& .\.venv\Scripts\python.exe .\tests\evaluate_examples.py
```
