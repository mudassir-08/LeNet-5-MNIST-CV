# LeNet-5-Style MNIST Classifier

A PyTorch implementation and deployment demo for handwritten-digit classification on MNIST. This project is my **starting point in machine-learning research**: a reproducible-in-progress baseline for learning how to formulate experiments, evaluate a model, analyze errors, and build on prior work. It does not propose a new architecture or claim a novel research contribution.

## Project status and research scope

The project currently provides a training and evaluation workflow in a Jupyter notebook, reusable model and helper modules, and a Gradio inference app. The experiment is best treated as an educational baseline and a foundation for future research questions—not as a benchmark study or production-ready system.

**Research question to build from:** How do model design and training choices affect classification performance and error patterns on handwritten digits?

## Results

The checked-in notebook contains recorded output from one 15-epoch training run:

| Split / metric | Result |
|---|---:|
| Training accuracy | 99.30% |
| Validation accuracy | 98.98% |
| Test accuracy | 98.82% |

These are the values recorded in the notebook, not independently reproduced results or averages across multiple runs. The notebook also displays training curves, sample predictions, and a test-set confusion matrix. To make the results suitable for rigorous comparison, future experiments should repeat runs across multiple seeds and report variability, document the exact environment, and compare against explicit baselines.

## Dataset and preprocessing

The experiment uses MNIST: 60,000 training images and 10,000 test images of handwritten digits in 10 classes. Images are resized to 32×32, converted to tensors, and normalized using mean 0.5 and standard deviation 0.5.

The notebook requests a 10% validation split from the 60,000-example training set. In `src/helper_dataset.py`, the split uses the first 54,000 indices for training and the last 6,000 for validation, with random sampling within each subset. This is a fixed index-based split, not a newly randomized or stratified split. The official test set is evaluated after training.

## Model

The model in `src/model.py` is a compact **LeNet-5-style variant** for grayscale input, with two convolutional layers, `tanh` activations, max pooling, and fully connected layers of 120 and 84 units followed by a 10-class output layer. The pooling and implementation details should be considered when comparing it with the original LeNet-5 architecture.

## Training and evaluation

The training notebook is `notebooks/LeNet5.ipynb`; reusable code is in `src/`:

- `model.py` — LeNet-5-style network definition.
- `helper_dataset.py` — MNIST loading and train/validation/test data loaders.
- `helper_train.py` — training loop using cross-entropy loss and the optimizer supplied by the notebook.
- `helper_evaluation.py` — accuracy, confusion-matrix, and seed utilities.
- `helper_plotting.py` — training curves, sample predictions, and confusion-matrix plots.

The recorded notebook run uses seed 123, batch size 256, 15 epochs, and SGD with a learning rate of 0.1. A `ReduceLROnPlateau` scheduler is created in the notebook but is not passed to the training function, so it does not affect the recorded run. The seed utility sets random seeds; the notebook does not call the separate deterministic-algorithms utility. Therefore, identical results across hardware and software environments are not guaranteed.

## Run the notebook

From the repository root, install the listed dependencies and Matplotlib (used by the plotting helpers), then open the notebook:

```bash
python -m pip install -r requirements.txt matplotlib
jupyter notebook notebooks/LeNet5.ipynb
```

The dependency versions are not pinned. The notebook was recorded in a Google Colab environment; results and runtime may vary in other environments. Training is notebook-based—there is no standalone training command in this repository.

## Inference demo

A deployed Gradio demo is available at [LeNet-5 MNIST Demo](https://huggingface.co/spaces/Mudassir-08/lenet5-mnist-demo). The app implementation is in `app.py`. It downloads `lenet5_model.pth` from the Hugging Face model repository configured in the app (`Mudassir-08/lenet5-mnist`) at startup and returns the predicted digit. The checkpoint is hosted externally and is not included in this repository. The demo is intended for MNIST-like, single-digit images; performance on arbitrary uploaded images is not established by the MNIST test accuracy.

To run the app locally, install the dependencies and launch it from the repository root:

```bash
python -m pip install -r requirements.txt
python app.py
```

A reachable model checkpoint at the configured Hugging Face repository is required for the app to load successfully.

## Limitations and next research steps

- This is a single MNIST experiment with one recorded run; no multi-seed uncertainty estimates are reported.
- There is no baseline comparison or ablation study, so the results do not isolate the effect of individual design choices.
- The fixed index-based validation split should be assessed or replaced with a documented randomized, stratified split for controlled comparisons.
- The current analysis emphasizes accuracy and visualizations; add per-class precision/recall, quantitative error analysis, and confidence calibration where relevant.
- Pin dependency versions and document hardware/software details to improve reproducibility.
- Keep deployment claims scoped to the interactive demo; robustness to non-MNIST images has not been evaluated.

These are opportunities to develop the project into a more rigorous research exercise. Any future claim of improvement should be backed by controlled comparisons and clearly reported evaluation procedures.

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 86(11), 2278–2324.
- MNIST database: [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/).
