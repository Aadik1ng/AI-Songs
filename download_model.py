from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_auth_token=None)
model = GPT2LMHeadModel.from_pretrained("distilgpt2", use_auth_token=None)

tokenizer.save_pretrained("./distilgpt2")
model.save_pretrained("./distilgpt2")
