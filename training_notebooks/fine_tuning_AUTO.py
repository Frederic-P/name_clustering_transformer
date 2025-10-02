# automatic fine_tuning script
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import sys

sys.path.append('../utils')
import databalancing
import gpu_manager
from logger import Logger


# ------------------------
# Setup
# ------------------------
models = [
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/LaBSE',
    'FacebookAI/xlm-roberta-base'
]
databalances = ['balanced', 'unbalanced']
patience = 4    # max epochs of no improved F1 before early stopping
seed = 42       # randomness control

log = Logger(base_dir= '../')
log.log('[INFO]: Script launched')
gpu_manager.pick_gpu(1, 'auto', 1)


# ------------------------
# Load + balance data
# ------------------------
data = pd.read_csv(r"../data/pairdata.csv")

train_df = data[data["dataset"] == "train"]
val_df = data[data["dataset"] == "validation"]
test_df = data[data["dataset"] == "test"]

ml_data = {}
ml_data['unbalanced'] = {
    'train_df': train_df,
    'val_df': val_df,
    'test_df': test_df
}

balanced_train_df = databalancing.rebalance_dataset(train_df, random_state=seed)
balanced_val_df = databalancing.rebalance_dataset(val_df, random_state=seed)
balanced_test_df = databalancing.rebalance_dataset(test_df, random_state=seed)
ml_data['balanced'] = {
    'train_df': balanced_train_df,
    'val_df': balanced_val_df,
    'test_df': balanced_test_df
}


# ------------------------
# Dataset
# ------------------------
class NamePairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=32):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text1, text2, label = str(row.name1), str(row.name2), int(row.are_same)
        encoding = self.tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ------------------------
# Metrics
# ------------------------
def ez_metrics(labels, preds):
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return ez_metrics(labels, preds)


# ------------------------
# Logging Callback
# ------------------------
class LoggingCallback(TrainerCallback):
    """Logs F1, LR, and params per epoch with Logger."""

    def __init__(self, logger, total_epochs, training_args):
        self.logger = logger
        self.total_epochs = total_epochs
        self.training_args = training_args

    def on_epoch_end(self, args, state, control, **kwargs):
        # Extract most recent eval log
        f1 = None
        lr = None
        if state.log_history:
            last_log = state.log_history[-1]
            f1 = last_log.get("eval_f1", None)
            lr = last_log.get("learning_rate", None)

        self.logger.log(
            message = "[EPOCH REPORT]:",
            object = {
                "epoch": state.epoch,
                "f1": f1,
                "learning_rate": lr,
                "train_batch_size": self.training_args.per_device_train_batch_size,
                "eval_batch_size": self.training_args.per_device_eval_batch_size,
                "weight_decay": self.training_args.weight_decay,
                "base_lr": self.training_args.learning_rate,
                "total_epochs": self.total_epochs,
            },
        )


# ------------------------
# Training args builder
# ------------------------
def build_training_args(experiment_name, patience=4, learning_rate=2e-5, num_train_epochs=10):
    return TrainingArguments(
        output_dir=f"./results_{experiment_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,  # keep last + best
        lr_scheduler_type="linear",  # linear decay
        logging_strategy="epoch",
        logging_dir=f"./logs_{experiment_name}",
        logging_first_step=True,
        report_to="none",  # no external reporting
        seed=seed,
    )


# ------------------------
# Training Loop
# ------------------------
i = 0
for modelname in models:
    for balance in databalances:
        i += 1
        experiment_name = modelname.split('/')[1] + f"_{balance}"
        log.log(f"[INFO] Training {i} of {len(models)*len(databalances)} {experiment_name}...")
        print(f"Training {i} of {len(models)*len(databalances)} {experiment_name}...")

        # Tokenizer + datasets
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        train_df = ml_data[balance]['train_df']
        val_df = ml_data[balance]['val_df']
        test_df = ml_data[balance]['test_df']
        train_dataset = NamePairDataset(train_df, tokenizer)
        val_dataset = NamePairDataset(val_df, tokenizer)
        test_dataset = NamePairDataset(test_df, tokenizer)

        # Model
        model = AutoModelForSequenceClassification.from_pretrained(modelname)

        # Training args
        training_args = build_training_args(
            experiment_name=experiment_name,
            patience=patience,
            learning_rate=2e-5,
            num_train_epochs=10
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                LoggingCallback(log, training_args.num_train_epochs, training_args),
                EarlyStoppingCallback(early_stopping_patience=patience)
            ]
        )

        # Train
        trainer.train()

        # Random guessing baseline
        actuals = test_df['are_same']
        preds = torch.randint(low=0, high=2, size=(len(actuals),))
        random_metrics = ez_metrics(actuals, preds)
        print("Random guessing baseline on test set:", random_metrics)

        # Evaluate
        metrics = trainer.evaluate(test_dataset)
        print("Test metrics:", metrics)

        log.log(message='[RESULT:] Random guessing results:', object=random_metrics)
        log.log(message='[RESULT:] Trained model results:', object=metrics)

        # Save best model
        save_dir = f"../completed_experiments/{experiment_name}"
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        log.log(message=f"[OUTPUT:] model saved to: {save_dir}")

log.log(message="Completed")
log.close()
