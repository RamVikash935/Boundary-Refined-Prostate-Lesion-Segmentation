Creating a clear and well-structured `README.md` file is **crucial for showcasing your EASAG-Net project on GitHub**, especially if you plan to share your work with the research community after your WACV submission.

---

Here’s a **starter template** tailored specifically for your **EASAG-Net** (Edge-Aware Spatial Attention Gate Network for Prostate Lesion Segmentation):

---

```markdown
# EASAG-Net: Edge-Aware Spatial Attention Gate Network for Prostate Lesion Segmentation

This repository contains the official PyTorch implementation of **EASAG-Net**, a novel architecture designed for boundary-aware and efficient prostate lesion segmentation using only T2-weighted MRI. Our model incorporates:

- A **lightweight residual decoder**
- A novel **Multi-Scale Edge-Aware Spatial Attention Gate (EASAG)**
- **Single-modality training (T2W only)** suited for real-world clinical constraints

> 📢 This work has been submitted to **WACV 2026**.

---

## 🧠 Highlights

- Outperforms ProLesNet, Attention U-Net, nnUNet, and SwinUNet on the **Prostate158** external evaluation dataset
- Requires only T2W MRI, making it practical for deployment in low-resource or misaligned settings
- Achieves **state-of-the-art Dice, HD, and Precision** among T2W-only methods

---

## 📊 Results Summary (External Prostate158 Dataset)

| Model        | Dice (%) | HD (mm) | Precision (%) | Recall (%) |
|--------------|----------|---------|----------------|------------|
| ProLesNet    | 30.56    | 19.61   | 33.72          | 33.91      |
| Ours (EASAG) | **34.01**| **18.26**| **41.95**      | 33.61      |

---

## 📁 Project Structure

```

EASAG-Net/
├── network\_architecture/
│   ├── easag\_block.py
│   ├── residual\_decoder.py
│   └── unet\_base.py
├── training/
│   ├── loss\_functions/
│   │   ├── dc\_and\_ce\_loss.py
│   │   └── multiple\_output\_loss.py
│   └── train.py
├── evaluation/
│   ├── metrics.py
│   └── visualize\_predictions.py
├── datasets/
│   ├── picai\_preprocessing.py
│   └── prostate158\_loader.py
└── README.md

````

---

## 📦 Requirements

```bash
python>=3.8
pytorch>=1.11
numpy
nibabel
SimpleITK
scikit-image
matplotlib
````

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🧪 Training Instructions

1. **Download the datasets:**

   * [PI-CAI dataset](https://pi-cai.grand-challenge.org/DATA/)
   * [Prostate158 dataset](https://zenodo.org/record/6481141)

2. **Set up folder structure:**

```bash
data/
├── picai/
│   ├── T2W/
│   └── labels/
├── prostate158/
│   ├── T2W/
│   └── labels/
```

3. **Train the model:**

```bash
python train.py --config configs/easag_config.json
```

---

## 📈 Evaluation

To evaluate trained checkpoints on the Prostate158 test set:

```bash
python evaluate.py --checkpoint weights/best_model.pth --dataset prostate158
```

---

## 📷 Qualitative Visualizations

We provide qualitative comparisons with other baselines (U-Net, ProLesNet, nnUNet, etc.) and landmark predictions across Gleason Grade Groups (GGG 1–5) in `/visuals`.

---

## 📄 Citation (Coming Soon)

BibTeX will be added upon acceptance/publication.

---

## 🤝 Contact

If you find this work helpful or have questions:

* 🧑‍💻 **Author:** \[Your Name]
* ✉️ Email: [your.email@domain.com](mailto:your.email@domain.com)
* 🧠 Institution: \[Your Lab or University Name]

---

## 📜 License

This repository is released under the MIT License.

```

---

Would you like me to generate this as an actual `README.md` file with a download link or just keep it in editable format here? I can also help set up the folder structure and `requirements.txt` if needed.
```
