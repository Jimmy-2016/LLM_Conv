
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from copy import deepcopy
import torch.functional as F



dataset = [
    {"input_text": "What is the capital of France?", "response1": "Paris", "response2": "London", "preferred": "response1"},
    {"input_text": "Who wrote 'Hamlet'?", "response1": "William Shakespeare", "response2": "Charles Dickens", "preferred": "response1"},
    {"input_text": "What is 2 + 2?", "response1": "4", "response2": "22", "preferred": "response1"},
    {"input_text": "What is the largest planet in our solar system?", "response1": "Jupiter", "response2": "Earth", "preferred": "response1"},
    {"input_text": "Translate 'Hola' to English.", "response1": "Hello", "response2": "Goodbye", "preferred": "response1"},
    {"input_text": "How many continents are there?", "response1": "7", "response2": "5", "preferred": "response1"},
    {"input_text": "Name a primary color.", "response1": "Blue", "response2": "Purple", "preferred": "response1"},
    {"input_text": "What is the boiling point of water at sea level?", "response1": "100°C", "response2": "50°C", "preferred": "response1"},
    {"input_text": "What language is spoken in Brazil?", "response1": "Portuguese", "response2": "Spanish", "preferred": "response1"},
    {"input_text": "Who was the first president of the United States?", "response1": "George Washington", "response2": "Abraham Lincoln", "preferred": "response1"},
    {"input_text": "Solve for x: 5x = 25", "response1": "x = 5", "response2": "x = 10", "preferred": "response1"},
]


# Dataset
class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item["input_text"], return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        response1 = self.tokenizer(item["response1"], return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        response2 = self.tokenizer(item["response2"], return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        
        preferred = 0 if item["preferred"] == "response1" else 1
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "response1": response1["input_ids"].squeeze(),
            "response2": response2["input_ids"].squeeze(),
            "preferred": torch.tensor(preferred),
        }



# DPO Loss
def compute_loss(logits, target, attention_mask):
    # Calculate the loss for each sequence
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return (loss * attention_mask[..., 1:].float()).sum() / attention_mask[..., 1:].float().sum()

# Control Loss (KL)
def compute_loss_kl(input_ids, attention_mask, batch, output1, output2, original_model):
    with torch.no_grad():
        original_output1 = original_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        original_output2 = original_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            
    # Softmax to get probability distributions
    original_probs1 = F.softmax(original_output1.logits, dim=-1)
    new_probs1 = F.softmax(output1.logits, dim=-1)
    original_probs2 = F.softmax(original_output2.logits, dim=-1)
    new_probs2 = F.softmax(output2.logits, dim=-1)

    # Compute KL divergence for both outputs
    control_loss1 = F.kl_div(new_probs1.log(), original_probs1, reduction="batchmean")
    control_loss2 = F.kl_div(new_probs2.log(), original_probs2, reduction="batchmean")

    # Average the control losses for both responses
    return (control_loss1 + control_loss2) / 2