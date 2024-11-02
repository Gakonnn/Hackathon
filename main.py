from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Загружаем токенизатор и модель для задач Question Answering
tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large')

# Загружаем датасет KazQAD
ds = load_dataset("Kyrmasch/sKQuAD")


# Предобработка данных
def preprocess_data(examples):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]

    # Токенизация
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Добавляем метки для начала и конца ответа
    start_positions = [context.find(answer['text'][0]) for context, answer in zip(contexts, answers)]
    end_positions = [start + len(answer['text'][0]) for start, answer in zip(start_positions, answers)]

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


# Применяем предобработку к датасету
tokenized_dataset = ds.map(preprocess_data, batched=True)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Определяем Trainer для обучения и валидации
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Обучение модели
trainer.train()

# Оценка модели
eval_results = trainer.evaluate(tokenized_dataset["test"])
print("Baseline F1 and EM:", eval_results)
