# Training Improvements Applied

## Date: November 15, 2024

### Changes Applied to `src/model/train_balanced.py`

---

## 1. Probabilistic Noise Augmentation

**Previous:** Noise applied to 100% of augmented samples

**Updated:** Noise applied to 70% of augmented samples (30% stay clean)

**Location:** Line 91

**Rationale:**
- Allows model to learn both clean and noisy patterns
- Better robustness to real-world variations
- Standard practice in data augmentation
- Reduces risk of model becoming too dependent on noisy data

---

## 2. Simplified CNN Architecture

**Previous Architecture:**
```
3 Convolutional Layers:
- Conv1: 1 → 64 channels, kernel (1, 25)
- Conv2: 64 → 128 channels, kernel (1, 15)
- Conv3: 128 → 256 channels, kernel (3, 3)

3 Fully Connected Layers:
- FC1: 10,752 → 512
- FC2: 512 → 256
- FC3: 256 → 11

Parameters: ~10-12 million
```

**Updated Architecture:**
```
2 Convolutional Layers (both with 2D kernels):
- Conv1: 1 → 32 channels, kernel (5, 5)
- Conv2: 32 → 64 channels, kernel (3, 3)

3 Fully Connected Layers:
- FC1: 3,136 → 256
- FC2: 256 → 128
- FC3: 128 → 11

Parameters: 856,907 (5-10x fewer than original!)
```

**Benefits:**
✅ **Faster Training:** ~2-3 hours instead of 3-4 hours
✅ **Less Overfitting:** Fewer parameters reduce overfitting risk
✅ **Better Generalization:** Simpler models often generalize better (Occam's Razor)
✅ **Sufficient Capacity:** 28×28 images with 10 classes don't need deep networks
✅ **Easier to Tune:** Fewer hyperparameters to optimize
✅ **Pure 2D Approach:** Both conv layers use 2D kernels (5×5 and 3×3) to capture spatial patterns in network traffic

---

## Dataset Status

**Classes:** 10 (P2P VPN removed)
- Training: 65,816 samples
- Validation: 14,103 samples
- Test: 14,104 samples
- **Total: 94,023 samples**

**Balancing Strategy:**
- ✅ Weighted Random Sampler (oversamples minorities)
- ✅ Weighted CrossEntropyLoss (penalizes majority errors)
- ✅ On-the-fly augmentation (different each epoch)

**Augmentation Techniques:**
- Random noise (±8 bytes, 70% probability)
- Random truncation (70-100% of flow, 50% probability)
- Small vertical shift (±1 packet, 30% probability)

---

## Expected Performance

**Training Time:** ~2-3 hours (on CPU)

**Expected Metrics:**
- Overall Accuracy: 88-92%
- Per-class F1-score: 0.80-0.95 for major classes
- Validation loss: Should stabilize around 0.3-0.5

**What to Watch For:**
1. Training/validation gap < 5% (indicates good generalization)
2. Loss decreasing steadily for first 15-20 epochs
3. Learning rate reductions at epochs ~10, ~20, ~30
4. Best model likely around epoch 40-50

---

## Training Command

```bash
cd /Users/kivanc/Desktop/AttentionNet
source venv/bin/activate
python src/model/train_balanced.py
```

---

## Output Location

Results saved to: `model_output_balanced/`
- `best_model.pth` - Best model weights
- `final_model.pth` - Final epoch weights
- `training_history.png` - Loss/accuracy curves
- `confusion_matrix.png` - Test set confusion matrix
- `classification_report.txt` - Detailed per-class metrics
- `training_config.json` - Hyperparameters used

---

## Post-Training Analysis

After training completes, check:
1. **Confusion Matrix:** Which classes are confused?
2. **Per-Class Metrics:** Are minority classes performing well?
3. **Training Curves:** Signs of overfitting?
4. **Best Epoch:** When did the best model occur?

If results are unsatisfactory (<85% accuracy), consider:
- Increasing dropout to 0.6
- Adding weight decay (currently 1e-5)
- Adjusting augmentation strength
- Collecting more data for weak classes

---

## Notes

- P2P (VPN) class removed from dataset (will collect more data later)
- Model still outputs 11 classes (0-10), class 8 just has no training data
- Using both weighted sampling AND weighted loss for maximum balance
- Simpler architecture should be sufficient for 28×28 network traffic images

Good luck with training! 🚀

