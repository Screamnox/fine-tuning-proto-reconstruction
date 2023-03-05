import torch
from torch.utils.data import DataLoader
from transformers import MBartTokenizer, MBartForConditionalGeneration, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm

"""--SETTINGS--"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
checkpoint = "facebook/mbart-large-cc25"
tokenizer = MBartTokenizer.from_pretrained(checkpoint)
model = MBartForConditionalGeneration.from_pretrained(checkpoint)
model.to(device)

"""--DATA PROCESSING--"""

# As a test to make sure I'm doing it right, 
# I'm using a modified database of the real Latin Reconstruction database of AB Antiquo.
# Obviously for true fine-tuning I would use the entire database and the tokenize function will be modified.
datasets = load_dataset("csv", data_files="Latin_reconstruction_database - Copy.csv", sep=",")

def tokenize_function(example):
    inputs = tokenizer(example['ITALIAN'], truncation=True)     # padding=True, max_length=32 ?
    targets = tokenizer(example['LATIN-CORRECT'], truncation=True)      # padding=True, max_length=32 ?
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'decoder_input_ids': targets['input_ids'],
        'decoder_attention_mask': targets['attention_mask'],
        'labels': targets['input_ids']
    }

tokenized_datasets = datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["ITALIAN", "LATIN-CORRECT"])
# Maybe instead of doing a dynamic padding, we could do a padding in the tokenize_function and set a maximum length.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets.set_format("torch")
#TODO: Save tokenized_datasets to avoid retokenizing it each time

# Instead of using sklearn or others, I manually split the database. 
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(3000))
val_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(3000, 3500))
test_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(3500, 3967))

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

#TODO: Correct batch error !
# I guess it comes from a structuring problem.
"""
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
"""

"""--TRAINING--"""
# I preferred to create a training without using `Trainer class`.

num_epochs = 10

optimizer = AdamW(model.parameters(), lr=1e-5)
# loss_fn = torch.nn.CrossEntropyLoss()

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss = loss_fn(outputs.logits, labels)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
"""TODO:
        # running_loss += loss.item()
        # preds = outputs.logits.argmax(dim=1)
        # correct_predictions += torch.sum(preds == labels)

    # epoch_loss = running_loss / len(train_dataloader)
    # epoch_acc = correct_predictions / len(train_data)

    # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
"""

"""TODO:
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    # test_loss = test_running_loss / len(test_dataloader)
    # test_acc = test_correct_predictions / len(test_data)

    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
"""

model.save_pretrained("trained_model")