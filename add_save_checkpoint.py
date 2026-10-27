import json

NOTEBOOK = "fake_news_detection_distilbert.ipynb"

with open(NOTEBOOK, "r", encoding="utf-8") as f:
    nb = json.load(f)

save_code = [
    "# =====================================================================\n",
    "# Save Model Weights & Tokenizer (Executed after the last epoch)\n",
    "# =====================================================================\n",
    "import os\n",
    "\n",
    "OUTPUT_DIR = \"distilbert_fake_news\"\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "# 1. Save Hugging Face model & tokenizer for Streamlit UI\n",
    "model.save_pretrained(OUTPUT_DIR)\n",
    "tokenizer.save_pretrained(OUTPUT_DIR)\n",
    "\n",
    "# 2. Save PyTorch state dictionary checkpoint\n",
    "torch.save({\n",
    "    \"epoch\": EPOCHS,\n",
    "    \"model_state_dict\": model.state_dict(),\n",
    "    \"best_val_loss\": best_val_loss,\n",
    "    \"history\": history\n",
    "}, os.path.join(OUTPUT_DIR, \"distilbert_weights.pt\"))\n",
    "\n",
    "print(f\"[SUCCESS] Model weights and tokenizer saved to './{OUTPUT_DIR}'\")\n",
    "\n",
    "# (Google Colab optional) Zip & download model weights:\n",
    "# !zip -r distilbert_fake_news.zip distilbert_fake_news/\n",
    "# from google.colab import files; files.download('distilbert_fake_news.zip')\n"
]

save_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "save_weights_after_last_epoch",
    "metadata": {
        "id": "save_weights_after_last_epoch"
    },
    "outputs": [],
    "source": save_code
}

# Find training cell (which has 'model.load_state_dict(best_state)' or id 'c59347ec')
insert_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "c59347ec":
        insert_idx = i
        break

if insert_idx == -1:
    insert_idx = 20

# Insert immediately after the training cell
nb["cells"].insert(insert_idx + 1, save_cell)

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Successfully added save code block at cell index {insert_idx + 1}!")
