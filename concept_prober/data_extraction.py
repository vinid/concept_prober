from tqdm import tqdm
from multiprocessing import Pool
from typing import List



def generate_data_from_json(json_data, *, label="seeds"):
    elements_to_find = json_data[label]["words"]
    elements2concept_dict = {k: v for k, v in
                             zip(json_data[label]["words"], json_data[label]["concepts"])}
    element_to_concept = (lambda x: elements2concept_dict[x])
    return elements_to_find, element_to_concept


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





