# BrickKilnSegNet
This project uses deep learning to segment brick kilns in 8-channel satellite imagery across India—where over 100,000 kilns contribute significantly to CO₂ emissions and air pollution. Accurate mapping supports environmental monitoring and policymaking. We trained U-Net++ models and achieved an F1 score of 0.70 across 8 classes using a combined Tversky and Cross Entropy loss.


## Model

We experimented with several segmentation models:

- **U-Net (random weights)**: Initial results were weak.
- **U-Net (ImageNet weights, partially frozen)**: Poor performance.
- **U-Net (ImageNet weights, fully unfrozen)**: Improved significantly.
- **U-Net++**: Delivered the best results.

The models were adapted to handle **8-channel satellite imagery**.

## Loss Function

We explored multiple loss functions to improve performance:

- **Weighted Cross Entropy**: High recall but poor precision.
- **Dice Loss**: Balanced precision/recall for a few classes, but missed others.
- **Tversky Loss**: Generalized Dice loss with more control over false negatives; helped predict top 3 frequent classes with better balance.

Finally, we **combined Tversky and Weighted Cross Entropy**, tuning the weights to get the best of both worlds. This resulted in:

- **Overall F1 Score**: 0.70
- **Per-Class F1 Scores**: ≥ 0.65 for all 8 classes

## Evaluation

Evaluation was performed after every epoch, including:

- **Per-class F1 scores** (to understand class-wise performance)
- **Binary F1 score** (brick kiln vs. background)
- **Weighted F1 score**
- **Object-based F1 score** (post-training)

These metrics helped guide model selection and tuning.

## Inference: Sliding Window Approach

To handle large, high-resolution satellite images, we used a **sliding window** approach:

- **Window size**: 256 × 256
- **Stride**: 128 pixels
- Overlapping patches were processed in batches.
- Class probabilities were averaged across overlapping regions to smooth predictions and reduce edge artifacts.

## Notes

- **Data Augmentation**: Tried, but it worsened validation F1 scores.
- **Classes**: The model was trained to segment 8 distinct classes of brick kilns.

---

