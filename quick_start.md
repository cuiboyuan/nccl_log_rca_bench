
Evaluate:
- DeepLog, LogAnomaly, LogBert:
    - `python evaluation/evaluate.py --model deeplog loganomaly`
    - LogBert has a bug at the moment (July 9)
- LLM-based:
    - `python evaluation/evaluate_logprompt.py --api-key <OPENAI_API_KEY>  --model gpt-5.4 --strategy CoT`
- Log-counting:
    - `python evaluation/allreduce_count_baseline.py`