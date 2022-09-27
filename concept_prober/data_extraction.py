from tqdm import tqdm
from multiprocessing import Pool
import os
from typing import List


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
            for k in matches:
                for key, value in k:
                    filino.write(f"{key}\t{value.strip()}\n")





