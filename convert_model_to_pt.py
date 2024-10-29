from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("./model/model")
tokenizer = AutoTokenizer.from_pretrained("./model/processor")

torch.save(model.state_dict(), "./model/model.pt")

model_config = model.config.to_dict()
torch.save(model_config, "./model/model_config.pt")