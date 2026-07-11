
Evaluate:
- Split and parse train/test data:
    - `python evaluation/data_process.py  --normal-train-ratio 0.5 --abnormal-train-ratio 0.8`
- DeepLog, LogAnomaly, LogBert:
    - `python evaluation/evaluate.py --model deeplog loganomaly --skip-preprocess`
    - LogBert has a bug at the moment (July 9)
- LLM-based:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --api-key <OPENAI_API_KEY>`
- Log-counting:
    - `python evaluation/allreduce_count_baseline.py`