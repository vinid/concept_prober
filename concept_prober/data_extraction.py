from tqdm import tqdm
from multiprocessing import Pool
import gc
import os
import shutil

class MatchExtraction:

    def __init__(self, stimuli):
        self.stimuli = stimuli

    def func(self, line):
        collected = []
        for word in self.stimuli:
            if word in line.split():
                collected.append((word, line))
        return collected

    def extract(self, sentences, output_folder, save_name, batch_size=10, cpus=4):

        isExist = os.path.exists(output_folder)

        if isExist:
            raise Exception("Folder already exists")
        else:
            os.makedirs(output_folder, exist_ok=False)

        print("Finding Matches")

        with Pool(cpus) as pool:
            matches = list(tqdm(pool.imap(self.func, sentences), total=batch_size, position=0))

        with open(f"{output_folder}/{save_name}.tsv", "w") as filino:
            for k in matches:
                for key, value in k:
                    filino.write(f"{key}\t{value.strip()}\n")





