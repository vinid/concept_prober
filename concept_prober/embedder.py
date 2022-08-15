from transformers import *
import pandas as pd
import datasets
from torch.utils.data import DataLoader


class Embedder:

    def __init__(self, model_name, max_length, device="cuda"):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device

    def find_sub_list(self, sl, l):
        results = list()
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind: ind + sll] == sl:
                results.append((ind, ind + sll))

        return results

    def subset_of_tokenized(self, list_of_tokens):
        idx = list_of_tokens.index(self.tokenizer.pad_token_id)
        return list_of_tokens[0:idx]

    def embed(self, target_texts, words, layer_id, batch_size, averaging=False):

        words = [f" {word.strip()}" for word in words]

        original = pd.DataFrame({"text": target_texts, "words": words})

        test_dataset = datasets.Dataset.from_pandas(original)

        def tokenizer_function(examples):
            text_inputs = self.tokenizer(examples["text"], max_length=self.max_length, padding="max_length", truncation=True)
            word_inputs = self.tokenizer(examples["words"], max_length=20, padding="max_length", truncation=True,
                                         add_special_tokens=False)

            examples["input_ids"] = text_inputs.input_ids
            examples["attention_mask"] = text_inputs.attention_mask
            examples["words_input_ids"] = word_inputs.input_ids

            return examples

        encoded_test = test_dataset.map(tokenizer_function, remove_columns=["text", "words"])
        encoded_test.set_format("pt")

        dl = DataLoader(encoded_test, batch_size=batch_size)

        embs = []

        for batch in dl:
            words_ids = batch["words_input_ids"]

            del batch["words_input_ids"]

            batch = {k: v.to(self.device) for k, v in batch.items()}

            features = self.model(**batch)["hidden_states"]

            layer_features = features[layer_id]

            try:

                idx = [
                    self.find_sub_list(self.subset_of_tokenized(tok_word.tolist()), input_ids.tolist())[0]
                    for tok_word, input_ids in zip(words_ids, batch["input_ids"])
                ]
            except IndexError as e:
                raise Exception("Index Error: do all the words occur in the respective sentences?")

            if averaging:
                for embedded_sentence_tokens, (l_idx, r_idx) in zip(layer_features, idx):
                    embs.append(embedded_sentence_tokens[l_idx:r_idx, :].mean(0).detach().cpu().numpy())
            else:
                for embedded_sentence_tokens, (l_idx, r_idx) in zip(layer_features, idx):
                    embs.append(embedded_sentence_tokens[l_idx:l_idx+1, :].mean(0).detach().cpu().numpy())

        return embs



