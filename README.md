# CIFAR-10 Optimizer Benchmark

This project benchmarks the performance of several optimizers (Adam, Muon, Scion, and Dion) on the CIFAR-10 dataset, using both ResNet-18 and a small Vision Transformer (ViT) model.

## Setup and Usage

To reproduce the results, follow these steps:

1.  **Clone the repository and install the dependencies:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    pip install -r requirements.txt
    ```

2.  **Run the training scripts:**

    The following commands were used to train the models. Each command trains a specific model with a specific optimizer for 80 epochs with mixed precision.

    ```bash
    # ResNet18 + Optimizers
    python training.py --model resnet --opt adam  --epochs 80 --amp --out_dir ./results/resnet_adam
    python training.py --model resnet --opt dion  --epochs 80 --amp --out_dir ./results/resnet_dion
    python training.py --model resnet --opt muon  --epochs 80 --amp --out_dir ./results/resnet_muon
    python training.py --model resnet --opt scion --epochs 80 --amp --out_dir ./results/resnet_scion

    # MiniViT (ViT-small for CIFAR) + Optimizers
    python training.py --model vit --opt adam  --epochs 80 --amp --out_dir ./results/vit_adam
    python training.py --model vit --opt dion  --epochs 80 --amp --out_dir ./results/vit_dion
    python training.py --model vit --opt muon  --epochs 80 --amp --out_dir ./results/vit_muon
    python training.py --model vit --opt scion --epochs 80 --amp --out_dir ./results/vit_scion
    ```

3.  **Generate the summary and plots:**

    Run the `main.ipynb` notebook to generate a summary of the results and the comparison plots.

## Results

The following plots show the training loss and validation accuracy versus the number of epochs for each optimizer and model.

**Training Loss vs. Epoch**

![Training Loss vs. Epoch](report/train_loss_vs_epoch.png)

**Validation Accuracy vs. Epoch**

![Validation Accuracy vs. Epoch](report/val_acc_vs_epoch.png)

## Conclusions

-   **Scion** demonstrates early-epoch acceleration and competitive final accuracy, especially on ResNet.
-   **Muon** shows a trade-off between more expensive updates and good conditioning, with expected gains in longer runs or larger models.
-   **Dion** is optimized for synchronous, communication-efficient training at scale and its performance is not fully exercised in this single-GPU setup.
-   **Adam** remains a strong general-purpose baseline, particularly for the small ViT.
