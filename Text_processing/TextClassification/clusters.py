
from Alignment.utils import *

# Load model and tokenizer
model_name = "bert-base-uncased"  # 
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('tokenized')

# Encode example texts

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Get average embedding
embeddings = embeddings.detach().numpy().squeeze()

print('emebedding is ready')
cluster_emb(embeddings, labels, method='PCA')
