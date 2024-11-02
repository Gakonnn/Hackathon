from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd



tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large')

ds = load_dataset("Kyrmasch/sKQuAD")

train_data = ds['train']

train_df = pd.DataFrame(train_data)

train_df, val_df = train_test_split(train_df, test_size=0.2)

train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)

def preprocess_data(examples):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answer"]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    start_positions = []
    end_positions = []

    for context, answer in zip(contexts, answers):
        if isinstance(answer, list) and len(answer) > 0:
            first_answer = answer[0]
            start = context.find(first_answer)
            if start != -1:
                start_positions.append(start)
                end_positions.append(start + len(first_answer))
            else:
                start_positions.append(-1)
                end_positions.append(-1)
        elif isinstance(answer, str):
            start = context.find(answer)
            if start != -1:
                start_positions.append(start)
                end_positions.append(start + len(answer))
            else:
                start_positions.append(-1)
                end_positions.append(-1)
        else:
            start_positions.append(-1)
            end_positions.append(-1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs



tokenized_train_dataset = train_data.map(preprocess_data, batched=True)
tokenized_val_dataset = val_data.map(preprocess_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

trainer.train()

eval_results = trainer.evaluate(tokenized_val_dataset)
print("Baseline F1 and EM:", eval_results)