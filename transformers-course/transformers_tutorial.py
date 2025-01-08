from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)   

raw_inputs = ["I love pizza.", "My cat won't stop miaowing."]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")

outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

def print_sentiment(tensor, statement):
    sentiment = ""
    if tensor[0] > tensor[1]:
        sentiment = "negative"
    else:
        sentiment  = "positive"
    print(f"'{statement}' is {sentiment} with a confidence of {torch.max(tensor).item()}")

for i in range(len(raw_inputs)):
    print_sentiment(predictions[i], raw_inputs[i])