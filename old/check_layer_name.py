
from transformers import AutoModelForSequenceClassification, AutoConfig
model_id = "Qwen/Qwen2-1.5B-Instruct"
config = AutoConfig.from_pretrained(model_id, num_labels=8)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_config(config)
print(model)
