from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd



# Загружаем токенизатор и модель для задач Question Answering
tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large')

# Загружаем датасет KazQAD
ds = load_dataset("Kyrmasch/sKQuAD")

# Получаем обучающий набор данных
train_data = ds['train']

# Преобразуем в DataFrame для использования train_test_split
train_df = pd.DataFrame(train_data)

# Разделяем на обучающую и валидационную выборки (например, 80% на обучение, 20% на валидацию)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Преобразуем обратно в Dataset
train_data = Dataset.from_pandas(train_df)
val_data = Dataset.from_pandas(val_df)

# Предобработка данных
def preprocess_data(examples):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answer"]

    # Токенизация
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Обработка ответов
    start_positions = []
    end_positions = []

    for context, answer in zip(contexts, answers):
        if isinstance(answer, list) and len(answer) > 0:
            # Если answer - это список, получаем первый ответ
            first_answer = answer[0]
            start = context.find(first_answer)
            if start != -1:  # Если ответ найден в контексте
                start_positions.append(start)
                end_positions.append(start + len(first_answer))
            else:
                # Если ответ не найден, добавьте -1 для обработки ошибок
                start_positions.append(-1)
                end_positions.append(-1)
        elif isinstance(answer, str):  # Если ответ - это строка
            start = context.find(answer)
            if start != -1:  # Если ответ найден в контексте
                start_positions.append(start)
                end_positions.append(start + len(answer))
            else:
                start_positions.append(-1)
                end_positions.append(-1)
        else:
            start_positions.append(-1)  # Если ответ отсутствует
            end_positions.append(-1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs



# Применяем предобработку к датасету
tokenized_train_dataset = train_data.map(preprocess_data, batched=True)
tokenized_val_dataset = val_data.map(preprocess_data, batched=True)

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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

# Обучение модели
trainer.train()

# Оценка модели
eval_results = trainer.evaluate(tokenized_val_dataset)
print("Baseline F1 and EM:", eval_results)
