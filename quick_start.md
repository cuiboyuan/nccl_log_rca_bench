Collect Data:
- Run script to collect data
- Put phase1_runs, phase2_runs, and labels in dataset/
- Run `python evaluation/group_by_second.py --dataset-dir dataset/`

Evaluate:
- Download the dataset:
    - `curl -LO https://github.com/cuiboyuan/nccl_log_rca_bench/releases/download/v0.1.0/nccl_log_benchmark.zip`
- Split and parse train/test data:
    - `python evaluation/data_process.py  --normal-train-ratio 0.5 --abnormal-train-ratio 0.5 --window-size-secs 5 --stride-secs 1`
- DeepLog, LogAnomaly, LogBert:
    - `python evaluation/evaluate.py --skip-preprocess`
- LLM-based, CoT strategy, default rules:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --api-key <OPENAI_API_KEY>`
- LLM-based, in-context strategy:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy InContext --skip-preprocess --example-file incontext_examples.csv --api-key <OPENAI_API_KEY>`
    - Jul 16 in context is breaking at the moment
- LLM-based, CoT strategy, custom rules:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --cot-rules-file nccl_cot_rules.txt --api-key <OPENAI_API_KEY>`
- LLM-based, CoT strategy, default rules, raw logs:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --log-type raw --api-key <OPENAI_API_KEY>`
- LLM-based, in-context strategy, raw logs:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy InContext --skip-preprocess --log-type raw --example-file incontext_examples.csv --api-key <OPENAI_API_KEY>`
    - Jul 16 in context is breaking at the moment
- LLM-based, CoT strategy, custom rules, raw logs:
    - `python evaluation/evaluate_logprompt.py --model gpt-5.4 --strategy CoT --skip-preprocess --log-type raw --cot-rules-file nccl_cot_rules.txt --api-key <OPENAI_API_KEY>`
