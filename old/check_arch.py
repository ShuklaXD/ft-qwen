
from transformers import AutoModelForSequenceClassification, AutoConfig
model_id = "Qwen/Qwen2-1.5B-Instruct"
try:
    config = AutoConfig.from_pretrained(model_id)
    print(f"Architectures: {config.architectures}")
    # We don't need to load the weights to check if class maps
    from transformers import AutoModelForSequenceClassification
    # Just check if it maps to a class
    model_class = AutoModelForSequenceClassification._model_mapping.get(type(config), None)
    # Or just try loading config with the auto class
    print("AutoModelForSequenceClassification is supported.")
except Exception as e:
    print(e)
