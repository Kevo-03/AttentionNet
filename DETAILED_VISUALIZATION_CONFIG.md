# Detailed Visualization Configuration Guide

This guide explains how to customize the detailed label visualizations.

## 📊 Multiple Figure Visualization

The visualization now **automatically splits** labels into multiple figures to keep images large and readable!

### Default Settings

```python
SAMPLES = 6               # Number of sample images per label (per row)
LABELS_PER_FIGURE = 10   # Number of labels per figure/window
```

### Example Output Structure

If you have 35 detailed labels, you'll get:
- **Part 1:** Labels 1-10 → `detailed_labels_grid_part1.png`
- **Part 2:** Labels 11-20 → `detailed_labels_grid_part2.png`
- **Part 3:** Labels 21-30 → `detailed_labels_grid_part3.png`
- **Part 4:** Labels 31-35 → `detailed_labels_grid_part4.png`

## ⚙️ Customization Options

### Option 1: Change Number of Samples Per Label

To show more or fewer images per label, edit either file:
- `src/preprocess/detailed_visualize_simple.py`
- `src/preprocess/detailed_visualize.py`

```python
SAMPLES = 6  # Change this number
```

Examples:
- `SAMPLES = 3` → Fewer samples, faster rendering
- `SAMPLES = 9` → More samples, wider figures
- `SAMPLES = 12` → Maximum detail

### Option 2: Change Labels Per Figure

To adjust how many labels appear in each figure:

```python
LABELS_PER_FIGURE = 10  # Change this number
```

Examples:
- `LABELS_PER_FIGURE = 5` → Larger images, more figures
- `LABELS_PER_FIGURE = 15` → Smaller images, fewer figures
- `LABELS_PER_FIGURE = 20` → Compact view

**Recommendation:** Keep between 5-15 for best readability

### Option 3: Adjust Figure Size

The figure height automatically adjusts, but you can customize:

```python
fig_height = max(n_classes_in_chunk * 1.5, 8)  # Multiplier controls spacing
```

Examples:
- `* 1.2` → More compact (less vertical space)
- `* 2.0` → More spacious (more vertical space)

### Option 4: Adjust Image Quality

Change DPI for higher/lower resolution:

```python
plt.savefig(output_file, dpi=150, bbox_inches='tight')  # Change dpi value
```

Examples:
- `dpi=100` → Faster, smaller files
- `dpi=200` → Higher quality, larger files
- `dpi=300` → Print quality

### Option 5: Change Colormap

To use a different color scheme:

```python
plt.imshow(class_imgs[idx], cmap="gray", ...)  # Change "gray"
```

Popular alternatives:
- `cmap="viridis"` → Colorful blue-green-yellow
- `cmap="plasma"` → Purple-orange
- `cmap="hot"` → Black-red-yellow
- `cmap="gray"` → Traditional grayscale (current)

## 📁 Files to Edit

### Simple Visualization (Standalone)
**File:** `src/preprocess/detailed_visualize_simple.py`
- Quick and simple
- Only creates grid visualizations
- Best for rapid iteration

**Edit lines 27-28:**
```python
SAMPLES = 6                # Line 27
LABELS_PER_FIGURE = 10     # Line 28
```

### Full Visualization Suite
**File:** `src/preprocess/detailed_visualize.py`
- Creates all visualizations (grids + charts)
- Part of comprehensive analysis

**Edit lines 44-45:**
```python
SAMPLES_PER_CLASS = 6      # Line 44
LABELS_PER_FIGURE = 10     # Line 45
```

## 🎨 Layout Examples

### Compact Layout (Many labels per figure)
```python
SAMPLES = 4
LABELS_PER_FIGURE = 15
```
Result: More labels visible at once, but smaller images

### Spacious Layout (Few labels per figure)
```python
SAMPLES = 9
LABELS_PER_FIGURE = 6
```
Result: Larger images, easier to see details, more figures

### Balanced Layout (Default)
```python
SAMPLES = 6
LABELS_PER_FIGURE = 10
```
Result: Good balance between visibility and number of figures

## 🖥️ Screen Size Recommendations

### Small Screen (13-14 inch laptop)
```python
SAMPLES = 4
LABELS_PER_FIGURE = 8
```

### Medium Screen (15-17 inch)
```python
SAMPLES = 6
LABELS_PER_FIGURE = 10  # Default
```

### Large Screen (24+ inch monitor)
```python
SAMPLES = 9
LABELS_PER_FIGURE = 15
```

### Ultra-wide Monitor
```python
SAMPLES = 12
LABELS_PER_FIGURE = 12
```

## 🚀 Quick Reference Commands

```bash
# Simple visualization (recommended for quick viewing)
python src/preprocess/detailed_visualize_simple.py

# Full visualization suite (all charts + grids)
python src/preprocess/detailed_visualize.py
```

## 📊 Output Files

### Simple Script Output
```
processed_test/visualizations/
├── detailed_labels_grid_part1.png
├── detailed_labels_grid_part2.png
├── detailed_labels_grid_part3.png
└── ...
```

### Full Script Output
```
processed_test/visualizations/
├── sample_images_grid_part1.png       # Grid visualization (split)
├── sample_images_grid_part2.png
├── sample_images_grid_part3.png
├── distribution_by_app_type.png       # Bar chart
├── distribution_by_app.png            # Bar chart
├── vpn_vs_nonvpn_comparison.png       # Grouped bars
├── heatmap_app_vs_apptype.png         # Heatmap
├── average_patterns_by_app_type.png   # Mean images
├── average_patterns_by_app.png        # Mean images
└── detailed_summary_report.txt        # Statistics
```

## 💡 Tips

1. **Start with defaults** - They work well for most cases
2. **Adjust LABELS_PER_FIGURE first** - Most impactful for readability
3. **Use simple script for iteration** - Much faster
4. **Increase DPI for presentations** - Use 200-300 for slides/papers
5. **Save originals** - Keep a backup before experimenting

## 🐛 Troubleshooting

### Images are too small
**Solution:** Decrease `LABELS_PER_FIGURE`
```python
LABELS_PER_FIGURE = 5  # Fewer labels = larger images
```

### Too many figure windows
**Solution:** Increase `LABELS_PER_FIGURE`
```python
LABELS_PER_FIGURE = 20  # More labels per figure
```

### Figures are too tall/wide
**Solution:** Adjust figure size multiplier
```python
fig_height = max(n_classes_in_chunk * 1.0, 8)  # Reduce multiplier
```

### Not enough samples shown
**Solution:** Increase `SAMPLES`
```python
SAMPLES = 9  # Show more samples per label
```

## 📝 Example Customization

For a **presentation-ready** visualization:

```python
# In detailed_visualize_simple.py or detailed_visualize.py

SAMPLES = 8                 # More samples for completeness
LABELS_PER_FIGURE = 6       # Fewer labels for clarity
# Change in savefig:
dpi=300                     # High resolution
```

Result: High-quality, large images perfect for presentations!

