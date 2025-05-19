import shutil
from transformers import TrainerCallback
from huggingface_hub import HfApi

class SaveEveryEpochCallback(TrainerCallback):
    def __init__(self, repo_name):
        self.repo_name = repo_name
        self.api = HfApi()

    def on_epoch_end(self, args, state, control, **kwargs):
        """Push model to Hugging Face Hub every epoch"""
        model = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)

        if model is None:
            print("⚠️ Model not found. Skipping save.")
            return

        epoch_num = round(state.epoch)
        branch_name = f"checkpoint-epoch-{epoch_num}"

        if epoch_num in [1, 5, 10, 13, 15]:
            # Save locally
            save_path = f"./{self.repo_name}-epoch-{epoch_num}"
            model.save_pretrained(save_path)
            if tokenizer:
                tokenizer.save_pretrained(save_path)

            # Ensure repo and branch exists
            self.api.create_repo(self.repo_name, private=True, exist_ok=True)
            branches = [b.name for b in self.api.list_repo_refs(self.repo_name).branches]
            if branch_name not in branches:
                self.api.create_branch(repo_id=self.repo_name, branch=branch_name)

            # Upload model to the new branch
            self.api.upload_folder(
                folder_path=save_path,
                repo_id=self.repo_name,
                repo_type="model",
                revision=branch_name
            )
            print(f"✅ Model pushed to Hugging Face Hub: {self.repo_name} (Epoch: {epoch_num}, Branch: {branch_name})")

            # Delete local checkpoint folder after successful upload
            shutil.rmtree(save_path, ignore_errors=True)
            print(f"🗑️ Deleted local checkpoint folder: {save_path}")
