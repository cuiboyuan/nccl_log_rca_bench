
Evaluate:
- Split and parse train/test data:
    - `python evaluation/data_process.py  --normal-train-ratio 0.8 --abnormal-train-ratio 0.5`
- DeepLog, LogAnomaly, LogBert:
    - `python evaluation/evaluate.py --skip-preprocess`
- LLM-based:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --api-key <OPENAI_API_KEY>`
- Log-counting:
    - `python evaluation/allreduce_count_baseline.py`