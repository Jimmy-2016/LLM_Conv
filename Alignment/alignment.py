from utils import *

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

original_model = deepcopy(model)
original_model.eval()

# Configure LoRA for the GPT-2 model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # Causal LM for GPT-2
    r=4,                          # Low-rank dimension
    lora_alpha=32,                # Scaling factor
    lora_dropout=0.1              # Dropout rate for LoRA
)
model = get_peft_model(model, lora_config)


# Create dataset and dataloader
dataset = PreferenceDataset(data=dataset, tokenizer=tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

num_epochs = 10
lambda_ = 0.9
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
training_loss = []
# Training
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass for response1
        output1 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["response1"])
        loss1 = compute_loss(output1.logits, batch["response1"], batch["attention_mask"])

        # Forward pass for response2
        output2 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["response2"])
        loss2 = compute_loss(output2.logits, batch["response2"], batch["attention_mask"])

        # Compute preference loss: log-likelihood difference
        preference_loss = (loss1 - loss2) * batch["preferred"].float() + (loss2 - loss1) * (1 - batch["preferred"].float())
        preference_loss = torch.relu(preference_loss).mean()

        # Control loss: KL divergence between original model and fine-tuned model for both outputs
        control_loss = compute_loss_kl(input_ids, attention_mask, batch, output1, output2, original_model)

        combined_loss = preference_loss + lambda_ * control_loss



        # Backward pass
        combined_loss.backward()
        optimizer.step()
        total_loss += combined_loss.item()
        training_loss.append(combined_loss.item())

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
