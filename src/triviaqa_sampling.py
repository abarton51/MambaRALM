import json
import random

def create_sample_triviaqa(full_dataset_path: str, sample_dataset_path: str, sample_size : int = 1000):

    #load full dataset
    with open(full_dataset_path, encoding="utf-8") as full_dataset_file:

        full_ds = json.loads(full_dataset_file.read())["Data"]

    sampled_full_ds = random.sample(full_ds, sample_size)

    with open(sample_dataset_path, "w", encoding="utf-8") as sample_dataset_file:

        sample_dataset_file.write(json.dumps(sampled_full_ds))

if __name__ == "__main__":

    create_sample_triviaqa("../data/triviaqa-rc/qa/wikipedia-dev.json", sample_dataset_path="../data/triviaqa-rc/qa/wikipedia-dev-sample.json")