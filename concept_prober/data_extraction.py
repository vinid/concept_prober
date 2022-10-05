from tqdm import tqdm
from multiprocessing import Pool
import os
from sklearn.metrics import classification_report
from typing import List


def combine_and_compute_performance(dataframe, instance_to_target):
    grouped_targets = dataframe.groupby(["word", "predictions"]).count().reset_index()
    sorted_targets = grouped_targets.sort_values(1, ascending=False).drop_duplicates(
        0).sort_index()  # drop "other" predictions
    test_y = [instance_to_target(k) for k in sorted_targets["word"]]

    return classification_report(test_y, sorted_targets["predictions"])


class FindWordTextOccurrence:

    def __init__(self):
        self.stimuli = None

    def func(self, line):
        collected = []
        for word in self.stimuli:
            if word in line.split():
                collected.append((word, line))
        return collected

    def extract(self, stimuli: List[str], sentences: List[str], output_location: str,  cpus=4):
        self.stimuli = stimuli

        with Pool(cpus) as pool:
            matches = list(tqdm(pool.imap(self.func, sentences), total=len(sentences), position=0))

        with open(f"{output_location}", "w") as filino:
            filino.write(f"word\tsentence\n")
            for k in matches:
                for key, value in k:
                    filino.write(f"{key}\t{value.strip()}\n")





