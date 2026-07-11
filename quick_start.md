
Evaluate:
- Download the dataset:
    - `curl -LO https://github.com/cuiboyuan/nccl_log_rca_bench/releases/download/v0.1.0/nccl_log_benchmark.zip`
- Split and parse train/test data:
    - `python evaluation/data_process.py  --normal-train-ratio 0.5 --abnormal-train-ratio 0.5`
- DeepLog, LogAnomaly, LogBert:
    - `python evaluation/evaluate.py --skip-preprocess`
- Log-counting:
    - `python evaluation/allreduce_count_baseline.py`
- LLM-based, CoT strategy, default rules:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --api-key <OPENAI_API_KEY>`
- LLM-based, in-context strategy:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy InContext --skip-preprocess --example-file incontext_examples.csv --api-key <OPENAI_API_KEY>`
- LLM-based, CoT strategy, default rules:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --api-key <OPENAI_API_KEY>`