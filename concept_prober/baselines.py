import numpy as np
from operator import itemgetter
from sklearn.metrics import classification_report
import pandas as pd


def cosine_all_average(seed_embeddings, instance_embeddings,
                       seed_dataframe: pd.DataFrame, instance_dataframe: pd.DataFrame,
                       instance_to_concept, layer_num=12):

    aggregated_seed_embeddings = {}

    for word in seed_dataframe["word"].unique().tolist():
        subset_of_ids = seed_dataframe[seed_dataframe["word"] == word].index.tolist()
        k = (itemgetter(*subset_of_ids)(seed_embeddings[layer_num]))

        if type(k) == tuple:
            aggregated_seed_embeddings[word] = np.average(k, axis=0)
        else:
            aggregated_seed_embeddings[word] = k

        aggregated_seed_embeddings[word] = np.average(k, axis=0)

    aggregated_instance_embeddings = {}

    for index, word in enumerate(instance_dataframe["word"].unique().tolist()):
        subset_of_ids = instance_dataframe[instance_dataframe["word"] == word].index.tolist()
        k = (itemgetter(*subset_of_ids)(instance_embeddings[layer_num]))
        if type(k) == tuple:
            aggregated_instance_embeddings[word] = np.average(k, axis=0)
        else:
            aggregated_instance_embeddings[word] = k

    mat_mul = np.array(list(aggregated_instance_embeddings.values())) @ np.array(list(aggregated_seed_embeddings.values())).T

    predicted_classes = np.argmax(mat_mul, axis=1).tolist()
    predictions = [seed_dataframe["word"].unique().tolist()[k] for k in predicted_classes]
    test_y = [instance_to_concept(k) for k in instance_dataframe["word"].unique().tolist()]
    return classification_report(test_y, predictions)


def cosine_baseline_average_concepts(seed_embeddings, instance_embeddings,
                                     seed_dataframe,
                                     instance_dataframe, instance_to_concept, layer_num=12):
    aggregated_seed_embeddings = {}

    for word in seed_dataframe["word"].unique().tolist():
        subset_of_ids = seed_dataframe[seed_dataframe["word"] == word].index.tolist()
        embeddings_from_ids = (itemgetter(*subset_of_ids)(seed_embeddings[layer_num]))

        aggregated_seed_embeddings[word] = np.average(embeddings_from_ids, axis=0)

    predicted_classes = np.argmax(instance_embeddings[layer_num] @ np.array(list(aggregated_seed_embeddings.values())).T, axis=1).tolist()
    instance_dataframe["cosine_predictions"] = [seed_dataframe["word"].unique().tolist()[k] for k in
                                                predicted_classes]

    group_targ = instance_dataframe.groupby(["word", "cosine_predictions"]).count().reset_index()
    sort_targ = group_targ.sort_values("sentence", ascending=False).drop_duplicates("word").sort_index()
    test_y = [instance_to_concept(k) for k in sort_targ["word"].values.tolist()]
    return classification_report(test_y, sort_targ["cosine_predictions"])


def cosine_baseline_no_averages(seed_embeddings,
                                instance_embeddings, seed_dataframe, instance_dataframe,
                                instance_to_concept, layer_num=12):
    pres = np.argmax((np.array(instance_embeddings[layer_num]) @ np.array(seed_embeddings[layer_num]).T), axis=1)

    instance_dataframe["cosine_predictions"] = seed_dataframe.iloc[pres]["word"].values
    group_targ = instance_dataframe.groupby(["word", "cosine_predictions"]).count().reset_index()
    sort_targ = group_targ.sort_values("sentence", ascending=False).drop_duplicates("word").sort_index()
    test_y = [instance_to_concept(k) for k in sort_targ["word"].values.tolist()]

    return classification_report(test_y, sort_targ["cosine_predictions"])
