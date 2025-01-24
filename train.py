from transformers import BertTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from dataset import Dataset
import os
import argparse
import warnings
import math
from fusion_embedding import FusionBertEmbeddings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PoemBERT training')
parser.add_argument("-mt", '--masking_type', type=str, default="CI-mask", choices=["CI-mask", "random", "span"])
parser.add_argument("--mask_decay", type=str, default="elliptical", choices=["linear", "cosine", "elliptic"])
args = parser.parse_args()

masking_type = args.masking_type
mask_decay = args.mask_decay

tokenizer = BertTokenizerFast.from_pretrained('./pretrained/bert-base-chinese')

if not os.path.exists("model"):
    os.makedirs("model")
model_path = "./model/" + masking_type + '_' + mask_decay

train_data = Dataset("./data", doc_len=512, mask_perc=0.3, masking_type=args.masking_type)

config = RobertaConfig(
    vocab_size=50265,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2
)


class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.roberta.embeddings = FusionBertEmbeddings(config)


model = CustomRobertaForMaskedLM(config=config)
print("Num param: ", model.num_parameters())

training_args = TrainingArguments(
    output_dir=model_path,
    overwrite_output_dir=True,
    num_train_epochs=80,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    prediction_loss_only=True,
    fp16=True
)


class MaskDecayCallback(TrainerCallback):
    def __init__(self, dataset, decay_type, total_epochs):
        self.dataset = dataset
        self.decay_type = decay_type
        self.total_epochs = total_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = state.epoch
        if self.decay_type == "linear":
            initial_mask_rate = 0.3
            new_mask_rate = initial_mask_rate * (1 - current_epoch / self.total_epochs)
        elif self.decay_type == "cosine":
            initial_mask_rate = 0.32
            final_mask_rate = 0.02
            new_mask_rate = final_mask_rate + 0.5 * (initial_mask_rate - final_mask_rate) * (
                        1 + math.cos(math.pi * current_epoch / self.total_epochs))
        elif self.decay_type == "elliptical":
            if current_epoch <= self.total_epochs / 2:
                new_mask_rate = 0.3 * math.sqrt(1 - (3 / (self.total_epochs ** 2)) * (current_epoch ** 2)) + 0.02
            else:
                new_mask_rate = 0.3 * (1 - math.sqrt(
                    1 - (3 / (self.total_epochs ** 2)) * ((self.total_epochs - current_epoch) ** 2))) + 0.02

        self.dataset.mask_perc = new_mask_rate
        print(f"Epoch {current_epoch}: Updated mask rate to {new_mask_rate:.4f}")


mask_decay_callback = MaskDecayCallback(
    dataset=train_data,
    decay_type=mask_decay,
    total_epochs=training_args.num_train_epochs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    callbacks=[mask_decay_callback]
)

trainer.train()
