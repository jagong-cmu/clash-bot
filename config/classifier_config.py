"""
Optional: custom path for the card classifier model.

If you leave MODEL_PATH as None, the program uses:
  image detector/card_classifier.pth
(which is where the trainer saves by default).

To use a different .pth file (e.g. a backup or a model trained elsewhere), set
MODEL_PATH to a full path or a path relative to the project root.
"""

# None = use default (image detector/card_classifier.pth)
MODEL_PATH = None

# Example: use a model in a different folder
# MODEL_PATH = "/Users/You/models/my_card_model.pth"

# Example: path relative to project root (script will resolve it)
# MODEL_PATH = "models/backup_classifier.pth"
