# Deep CoverageNet: Automated NFL Coverage Shell Classification

**Final Project - Deep Learning for Sports Analytics**

A deep learning system that predicts defensive coverage shells from early pre-snap movement using NFL tracking data. Achieves 87% accuracy at 90% confidence threshold, enabling selective auto-tagging for 22% of plays while reducing manual charting workload.

## 🏈 Project Overview

Deep CoverageNet classifies defensive coverage into six shells (2 MAN, C0, C1, C2, C3, C4) using only the first 1.2 seconds (12 frames) of pre-throw defender movement. The system combines automation benefits with strategic insights into defensive disguise quality.

## 🎯 Key Results

- **Overall Accuracy**: 66.2% across all plays
- **Macro F1 Score**: 0.536
- **High-Confidence Performance**: 87% accuracy at 90% confidence threshold
- **Coverage at Threshold**: Auto-tags 22% of plays with high precision
- **Practical Impact**: Reduces analyst workload by 40% at 80% confidence threshold

## 📊 Model Architecture Comparison

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 0.447 | 0.376 | Baseline |
| Random Forest | 0.547 | 0.447 | Classical baseline |
| **CNN** | **0.662** | **0.536** | Best overall |
| GRU | 0.586 | 0.514 | Strong temporal modeling |
| Transformer | 0.628 | 0.523 | High expressiveness |

**Winner: Convolutional Neural Network (CNN)**
- Captures spatial structure effectively
- Best balance of accuracy and calibration
- Robust performance across all coverage types

## 🔬 Methodology

### Data Processing

**Input Shape**: `[12 frames × 11 defenders × 8 features]`

**Features**:
- Relative positions (x, y)
- Velocities (vx, vy)
- Speed magnitude
- Movement deltas
- Standardized positional offsets

**Preprocessing**:
1. Align all plays to line of scrimmage
2. Normalize coordinate systems
3. Sort defenders consistently
4. Handle class imbalance through stratified splits

### CNN Architecture
```
Input: 12×11×8 tensor
Conv2D(32 filters, 3×3) → ReLU → BatchNorm
Conv2D(64 filters, 3×3) → ReLU → BatchNorm → MaxPool
Conv2D(128 filters, 3×3) → ReLU → BatchNorm → MaxPool
Flatten → Dense(256) → Dropout(0.3)
Output: Softmax(6 classes)
```

**Training**:
- Loss: Cross-entropy with class weights
- Optimizer: Adam (lr=0.001, weight decay=1e-4)
- Early stopping with patience=10
- Batch size: 32

## 📈 Per-Class Performance

| Coverage | Precision | Recall | F1 Score | Support |
|----------|-----------|--------|----------|---------|
| 2 MAN | 0.38 | 0.12 | 0.19 | 48 |
| C0 | 0.63 | 0.47 | 0.54 | 147 |
| C1 | 0.63 | 0.52 | 0.57 | 590 |
| C2 | 0.68 | 0.62 | 0.65 | 412 |
| C3 | 0.64 | 0.67 | 0.66 | 959 |
| C4 | 0.64 | 0.52 | 0.57 | 455 |

**Key Observations**:
- C2 and C3 show strongest performance (consistent early cues)
- C1/C3 confusion reflects real football ambiguity
- C4 often mislabeled as C3 (identical early leverage)
- 2 MAN weakest due to low support and overlap with C1

## 🎯 Confidence Calibration

The model demonstrates strong uncertainty awareness:

| Confidence Threshold | Accuracy | Macro F1 | Coverage |
|---------------------|----------|----------|----------|
| 0.50 | 0.662 | 0.536 | 86.7% |
| 0.60 | 0.729 | 0.605 | 60.2% |
| 0.70 | 0.755 | 0.631 | 43.8% |
| 0.80 | 0.803 | 0.656 | 39.5% |
| 0.90 | 0.870 | 0.714 | 21.7% |

**Calibration Quality**:
- Correct predictions cluster at 0.9-1.0 confidence
- Incorrect predictions cluster at 0.5-0.7 confidence
- Strong separation enables selective prediction workflow

## 💼 Business Applications

### 1. Automated Charting Workflow
```
For each play:
  if model_confidence >= 0.80:
    auto_tag(play)
  else:
    route_to_analyst(play)
```

**Impact**: 40% labor reduction at 80% accuracy

### 2. Coverage Disguise Index

Model uncertainty quantifies defensive deception:
- Aggregate confidence by team/shell/game
- Identify which defenses hold structure longest
- Measure QB exploitation of predictable coverages

**Strategic Value**:
- Benchmark own disguise effectiveness
- Scout opponent tendencies
- Identify vulnerable QBs vs disguise

### 3. Downstream Analytics

- Automated cut-up generation by coverage
- xEPA modeling by shell type
- Rotation detection and timing analysis
- Matchup preparation insights

## 🔍 Error Analysis

### Common Confusions

**C1 ↔ C3**: Most frequent error (164 cases)
- Both use single-high safety initially
- Differentiation requires rotation observation

**C4 → C3**: Second most common (140 cases)  
- Identical two-high leverage early
- Quarters rotation happens after 12 frames

**2 MAN dispersion**: Scattered predictions
- Only 48 examples in dataset
- Overlaps geometrically with multiple shells

### Football Validity

Model errors align with real defensive strategy:
- C3/C4 designed to look identical pre-snap
- 2 MAN/C1 share safety positioning until declaration
- Confusions reveal successful disguise, not model failure
- 
## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

### Running the Analysis
```python
# Load notebook
jupyter notebook notebooks/Deep_CoverageNet_Main.ipynb

# Training pipeline:
# 1. Load and preprocess tracking data
# 2. Create tensor representations (12×11×8)
# 3. Train CNN with class balancing
# 4. Evaluate on validation set
# 5. Analyze confidence calibration
# 6. Generate visualizations
```

### Data Requirements

- **Tracking Data**: NFL Next Gen Stats pre-throw defender movement
- **Labels**: Coverage shell classifications (6 categories)
- **Format**: CSV with frame-level coordinates, velocities, and metadata
- **Sample Size**: 13,611 plays (80/20 train/val split)

## 🎓 Technical Details

### Implementation

- **Framework**: Pure PyTorch (no high-level wrappers)
- **Custom Components**:
  - Manual training loops
  - Custom batching logic
  - Manual softmax extraction
  - Custom early stopping criteria

### Challenges Solved

1. **Defender Ordering**: Enforced consistent sorting by position
2. **Tensor Shapes**: Managed different architecture requirements
3. **Class Imbalance**: Applied class weights without degrading calibration
4. **Memory Management**: Optimized batch sizes for Transformer

### Model Selection Criteria

- **Accuracy**: Overall correctness
- **Macro F1**: Balanced performance across classes
- **Calibration**: Confidence reliability
- **Interpretability**: Error patterns match football logic

## 🔮 Future Enhancements

### Technical Improvements

- **Extended Frame Window**: Use 15-20 frames to capture more rotation
- **Attention Mechanisms**: Add interpretability via attention visualization
- **Class-Balanced Loss**: Further improve minority class performance
- **Ensemble Methods**: Combine CNN + Transformer predictions

### Feature Engineering

- **Safety Depth Metrics**: Explicit high/deep positioning features
- **Leverage Indicators**: Compute apex alignment explicitly
- **Motion Cues**: Track pre-snap motion and shifts
- **Formation Context**: Include offensive formation as input

### Deployment

- **Real-Time Inference**: Optimize for production environments
- **API Integration**: Connect to existing charting systems
- **Active Learning**: Flag uncertain cases for analyst feedback
- **Continuous Retraining**: Update with new seasons and schemes

## 📊 Coverage Distribution

Dataset composition:
- C3 (Cover 3): 3,200 plays (most common)
- C1 (Cover 1): 2,400 plays
- C4 (Cover 4): 2,400 plays
- C2 (Cover 2): 1,900 plays
- C0 (Cover 0): 700 plays
- 2 MAN (2-Man): 200 plays (rarest)

**Challenge**: Severe imbalance requires careful handling

## 🏆 Academic Context

**Course**: Deep Learning for Sports Analytics  
**Team**: Conor Patten, John Zhang, Stacy Chen, Emma Zhu, Tanmay Arya  
**Institution**: [Your University]  
**Semester**: [Term/Year]

**Deliverables**:
- Jupyter notebook with full implementation
- Technical report (10 pages)
- Presentation slides
- Model checkpoints and visualizations

## 👥 Team Contributions

**Conor Patten** (Primary Developer):
- Complete modeling framework design
- Preprocessing pipeline (alignment, normalization, tensors)
- All model implementations (LogReg, RF, CNN, GRU, Transformer)
- Training loops and evaluation functions
- Hyperparameter tuning and ablation studies
- Error analysis and calibration studies
- Visualization generation
- Report authoring (technical sections, results, insights)

**John Zhang**: Dataset exploration, label verification, edge case checking

**Stacy Chen**: Visualization review, report editing, figure organization

**Emma Zhu**: Literature review, introduction structuring, proofreading

**Tanmay Arya**: Output validation, results formatting, presentation feedback

## 📝 Citation
```bibtex
@misc{deepcoveragenet2024,
  title={Deep CoverageNet: Automated NFL Coverage Shell Classification},
  author={Patten, Conor and Zhang, John and Chen, Stacy and Zhu, Emma and Arya, Tanmay},
  year={2024},
  howpublished={Course Project},
  institution={[Your University]}
}
```

## 📧 Contact

**Lead Developer**: Conor Patten  
**Email**: conor.patten@duke.edu 
**LinkedIn**: https://www.linkedin.com/in/conor-patten/


## 🙏 Acknowledgments

- NFL and AWS for Next Gen Stats tracking data
- Course instructors and teaching staff
- Team members for collaboration and feedback
- Big Data Bowl community for data access

---

**Built with**: PyTorch • NumPy • scikit-learn • Matplotlib • Seaborn

**Status**: ✅ Project Complete

**Key Innovation**: First automated system to quantify defensive disguise quality through model uncertainty
