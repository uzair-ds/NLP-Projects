

# Textual Entailment Model

## Overview
This repository contains a **textual entailment model** fine-tuned on **DistilBERT (distilbert-base-cased)** to determine the logical relationship between sentence pairs.

## How to change the number of inputs?
Normally with transfer learning . we only change the head of the neural network . whole keeping the input + middle layer the same. But how to change the number of inputs? No need to change it. We can train the transformer to understand the multiple input sentences concatenated into the same input. This works with RNN as well

Format of input text : "[CLS] Some Text ABC. [SEP] Another text statement. [SEP]"

## What is Textual Entailment?
Textual entailment is an NLP task that classifies a **hypothesis** as:
- **Entailment**: Follows from the **premise**.
- **Contradiction**: Opposes the **premise**.
- **Neutral**: Unrelated to the **premise**.

## Fine-Tuning Process
- **Model**: DistilBERT (`distilbert-base-cased`)
- **Dataset**: SNLI, MNLI, or custom NLI data
- **Training**: Hugging Face Transformers, PyTorch/TensorFlow
- **Evaluation**: Accuracy, F1-score, loss



### Run Inference
```python
from model import predict_entailment
print(predict_entailment("A man is playing guitar.", "A person is making music."))
```

## Future Work
- Improve generalization with **domain adaptation**.
- Experiment with **larger transformer models**.
- Deploy as a **web API**.

## License
MIT License.

---
Contributions welcome!


