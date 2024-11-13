
from utils import *

from datasets import load_dataset

# Load DailyDialog dataset
dataset = load_dataset('daily_dialog', trust_remote_code=True)
# print(dataset)
# print(dataset['train'].features)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Step 1: Read conversation from a text file

# conversation = read_conversation('conv.txt')

dialogue, emotion = get_dialog(dataset, 100)

embeddings = [get_embeddings(sentense, tokenizer, model).detach().cpu().numpy() for sentense in dialogue]

embeddings = np.array(embeddings).squeeze()
emotions = np.array(emotion).squeeze()



cluster_emb(embeddings, emotions, method='PCA')
## visulize



