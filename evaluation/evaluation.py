from datasets import load_dataset

def load_evaluation_dataset():
    dataset = load_dataset("bryancui/nccl-log-rca-bench", split="train")
    return dataset

if __name__ == "__main__":
    dataset = load_evaluation_dataset()
    print(dataset[0])