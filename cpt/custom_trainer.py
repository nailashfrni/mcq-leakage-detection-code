from transformers import Trainer
from torch.optim import AdamW

class CustomPeftTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            lm_head_params = []
            other_params = []

            for name, param in self.model.named_parameters():
                if "lm_head" in name:
                    lm_head_params.append(param)
                else:
                    other_params.append(param)

            optimizer_grouped_parameters = [
                # {"params": embedding_params, "lr": 1e-5},  # Smaller LR for embeddings
                {"params": lm_head_params, "lr": 1e-5},    # Lower LR for `lm_head`
                {"params": other_params, "lr": 5e-4},      # Normal LR for other layers
            ]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                weight_decay=self.args.weight_decay
            )

        return self.optimizer