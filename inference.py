import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model import TransformerModel
import torch
from transformers import AutoTokenizer
import preprocess
import random
class Inference:
    def __init__(self, num_labels, model_path="intent_model.pt", model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TransformerModel(num_labels=num_labels)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        _, _, intent_response_dict, label_map = preprocess.PreProcessData(64).parse_json()
        self.idx_to_labels = {k: v for v, k in label_map.items()}
        self.responses = intent_response_dict
    def get_prediction(self, text):
        tokenized_text = self.tokenizer(text, return_tensors = "pt", padding = True, truncation = True)
        logits = self.model(**tokenized_text)
        softmax = torch.softmax(logits, dim=1)
        confidence = torch.max(softmax).item()
        if confidence < 0.6:
            raise RuntimeError("Confidence low, please try again")
        prediction = torch.argmax(logits, dim=1).item()
        return prediction
    def get_response(self, prediction):
        tag = self.idx_to_labels[prediction]
        responses_list = self.responses[tag]
        chosen_response = random.choice(responses_list)

        print(chosen_response)





infer = Inference(8)
prediction = infer.get_prediction("YO WHAT UP GANG")
infer.get_response(prediction)
