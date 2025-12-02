from transformers import DistilBertTokenizerFast
import json
import os


class PreProcessData:
    def __init__(self, max_length, data_path = "data.json", model_name = "distilbert-base-uncased"):

        if max_length > 512:
            raise ValueError("Max length cannot exceed 512 for distilbert")
        self.max_length = max_length

        self.data_path = data_path

        self.model_name = model_name

        if not os.path.exists(self.data_path):
            raise FileNotFoundError("File does not exist")
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
            if not isinstance(self.data["intents"], list):
                raise RuntimeError("JSON must contain a list of intents")
            for i, intent in enumerate(self.data["intents"]):
                if "tag" not in intent:
                    raise RuntimeError(f"Intent at index {i} must have a tag")
                if "patterns" not in intent or not isinstance(intent["patterns"], list):
                    raise RuntimeError(f"Intent {intent.get('tag', '<unknown>')} must have a list of patterns")
                if "responses" not in intent or not isinstance(intent["responses"], list):
                    raise RuntimeError(f"Intent {intent.get('tag', '<unknown>')} must have a list of responses")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)





    def tokenize(self, texts):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encodings

    def save_tokenizer(self, path="tokenizer_dir"):
        self.tokenizer.save_pretrained(path)

    def parse_json(self):
        intent_pattern_dict = {
            intent["tag"] : intent["patterns"]
            for intent in self.data["intents"]
        }

        intent_response_dict = {
            intent["tag"]: intent["responses"]
            for intent in self.data["intents"]
        }

        texts = [p for tag, patterns in intent_pattern_dict.items() for p in patterns]
        labels = [tag for tag, patterns in intent_pattern_dict.items() for _ in patterns]

        if len(intent_pattern_dict) != len(self.data["intents"]):
            raise ValueError("Duplicate intent found")
        if not texts:
            raise ValueError("No training patterns found")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("Patterns must be in the form of strings")

        label_map = {tag: idx for idx, tag in enumerate(intent_pattern_dict.keys())}
        labels_to_idx = [label_map[tag] for tag in labels]


        return texts, labels_to_idx, intent_response_dict, label_map



