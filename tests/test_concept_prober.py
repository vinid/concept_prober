import pytest
from concept_prober.data_extraction import MatchExtraction

import os




def test_extractor():
    matcher = MatchExtraction(["cat"])

    matcher.extract(["the cat is on the table", "a cat in my lap"], "temp_folder", "temp_name")
