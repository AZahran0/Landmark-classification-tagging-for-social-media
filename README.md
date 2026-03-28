# 🗺️ Landmark Classification Project

## 🔗 Live Demo
**[View the Live App on Hugging Face Spaces](https://huggingface.co/spaces/Zahran0/Landmark_Classification)**

## 📌 Description
This project is part of the **AWS Machine Learning Engineer Nanodegree Program** offered by **Udacity** and **AWS**. It focuses on building and improving Convolutional Neural Networks (CNNs) for landmark classification.

In photo-sharing services, metadata like GPS location helps organize and tag images. But many images lack this metadata. This project aims to solve that by classifying landmarks in images to infer their locations.

  ---
  ## Usage
  ### ⚙️ Requirements
  Make sure you have the following installed:
  ```bash
  pip install -r requirements.txt
  ```

  ### 1- Run as a Streamlit App
  ```bash
  streamlit run streamlit_app.py
  ```
  - Upload an image to:

    * Compare predictions from CNN, CNN + Residual Connections, and ResNet34.

    * View the Top-5 predicted classes with bar charts.
      <img width="1919" height="826" alt="image" src="https://github.com/user-attachments/assets/439bb41e-8058-4d8f-b08b-806b2082e2c3" />


  ### 2- Run as a Jupyter Notebook App (Voila)
  Launch the notebook as a standalone web app:
  ```bash
  voila app.ipynb
  ```

## 🧠 Project Overview
The goal was to train a model to identify landmarks in images. The project required building CNNs from scratch, applying transfer learning, and improving model performance.

### 💡 My standout contributions:
- Built a CNN from scratch for classification.
- Enhanced performance with **residual connections**.
- Applied **transfer learning using ResNet34**.
- Used **Grad-CAM** for visual model explainability.

---

## 📂 Dataset

- **Source**: A curated subset of the [Google Landmarks Dataset](https://github.com/cvdfoundation/google-landmark) provided by Udacity as part of the AWS Machine Learning Engineer Nanodegree program.
- **Classes**: [50 world landmarks](https://github.com/GLGL0x00/Landmark-classification-tagging-for-social-media/blob/main/Class_names.txt)
- **Images**: ~100 images per class (~5,000 total for training and 1250 images for test - ***for learning purposes***)
### 🧪 Preprocessing & Augmentation

Data augmentation pipeline was used to simulate real-world image variations and improve model robustness:

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```
💡 **Why this pipeline?**

- Input Size (224×224): Matches PyTorch pre-trained model expectations.

- Affine & Flip: Handle different angles, distances, and mirrored images.

- ColorJitter: Simulate lighting/camera variations.

- Blur: Mimic low-quality or out-of-focus images.

- RandomResizedCrop: Emulate zooming and framing differences.

---

## 🏋️ Model Training
To explore different strategies in landmark classification, I trained and compared three distinct models:

<details>
  <summary><strong>1- CNN From Scratch Architecture Summary</strong>
    <blockquote>
      <strong>Model Flow:</strong> Input → [Conv-BN-ReLU + MaxPool] ×5 → GAP → FC(50)
    </blockquote>
  </summary>

  <br>

  <table>
    <thead>
      <tr>
        <th>Stage</th>
        <th>Layers</th>
        <th>Output Shape</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Input</strong></td>
        <td>—</td>
        <td><code>[3, 224, 224]</code></td>
      </tr>
      <tr>
        <td><strong>Block 1</strong></td>
        <td>Conv(64) → BN → ReLU → MaxPool</td>
        <td><code>[64, 112, 112]</code></td>
      </tr>
      <tr>
        <td><strong>Block 2</strong></td>
        <td>Conv(128) → BN → ReLU → MaxPool</td>
        <td><code>[128, 56, 56]</code></td>
      </tr>
      <tr>
        <td><strong>Block 3</strong></td>
        <td>Conv(256) ×2 → BN → ReLU → MaxPool</td>
        <td><code>[256, 28, 28]</code></td>
      </tr>
      <tr>
        <td><strong>Block 4</strong></td>
        <td>Conv(512) ×2 → BN → ReLU → MaxPool</td>
        <td><code>[512, 14, 14]</code></td>
      </tr>
      <tr>
        <td><strong>Block 5</strong></td>
        <td>Conv(512) ×2 → BN → ReLU → MaxPool</td>
        <td><code>[512, 7, 7]</code></td>
      </tr>
      <tr>
        <td><strong>Head</strong></td>
        <td>GAP → Flatten → Dropout → FC(50)</td>
        <td><code>[50]</code></td>
      </tr>
    </tbody>
  </table>

</details>



  
<details>
  <summary><strong>2- CNN + Residual Connections Architecture Summary</strong>
    <blockquote><strong>Model Flow:</strong> Input → [ResBlock + MaxPool] ×5 → GAP → FC(50)</blockquote>
  </summary>
  <table>
    <thead>
      <tr>
        <th>Stage</th>
        <th>Layers</th>
        <th>Output Shape</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Input</strong></td>
        <td>—</td>
        <td><code>[3, 224, 224]</code></td>
      </tr>
      <tr>
        <td><strong>Block 1</strong></td>
        <td>ResidualBlock(3→64)<br>→ Conv-BN-ReLU ×2 + SkipConv (1×1) + ReLU<br>+ MaxPool(2×2)</td>
        <td><code>[64, 112, 112]</code></td>
      </tr>
      <tr>
        <td><strong>Block 2</strong></td>
        <td>ResidualBlock(64→128) + MaxPool</td>
        <td><code>[128, 56, 56]</code></td>
      </tr>
      <tr>
        <td><strong>Block 3</strong></td>
        <td>ResidualBlock(128→256) ×2 + MaxPool</td>
        <td><code>[256, 28, 28]</code></td>
      </tr>
      <tr>
        <td><strong>Block 4</strong></td>
        <td>ResidualBlock(256→512) ×2 + MaxPool</td>
        <td><code>[512, 14, 14]</code></td>
      </tr>
      <tr>
        <td><strong>Block 5</strong></td>
        <td>ResidualBlock(512→512) ×2 + MaxPool</td>
        <td><code>[512, 7, 7]</code></td>
      </tr>
      <tr>
        <td><strong>Block 6</strong></td>
        <td>ResidualBlock(512→512) ×2 + MaxPool</td>
        <td><code>[512, 7, 7]</code></td>
      </tr>
      <tr>
        <td><strong>Head</strong></td>
        <td>GlobalAvgPool → Flatten → Dropout(0.5) → Linear(512→50)</td>
        <td><code>[50]</code></td>
      </tr>
    </tbody>
  </table>

</details>

<strong>3. Transfer Learning using ResNet34<strong>
   - Pre-trained on ImageNet
   - Fine-tuned final layers for landmark classification


### 📈 Model Comparison Summary
**Hyperparameters:**
- batch_size => 64  
- num_epochs => 100      
- dropout => 0.4          
- learning_rate => 0.0001  
- optimizer => 'adam'          
- weight_decay => 0.001   


| Model                       | F1-score | Notes                          |
|-----------------------------|----------|--------------------------------|
| CNN from Scratch            | 70.88 %  | Baseline model                 |
| CNN + Residual Connections  | 74.8 %   | Residuals improved accuracy    |
| Transfer Learning (ResNet34)| 72  %    | Trained only for 50 epochs     |

#### 📝 Training Insights
- Switching the learning rate scheduler from **ExponentialLR** to **ReduceLROnPlateau** significantly boosted test accuracy (from **56% → 70%**).  
- Adding **residual connections** to the scratch CNN improved F1-score from **70% → 74%**

## 📊 Evaluation & Results

### ✅ Classification Report visualization
<details><summary><strong>CNN from Scratch</strong></summary>
  <img width="1990" height="1589" alt="CNN from Scratch visualization" src="https://github.com/user-attachments/assets/c8237479-7772-48be-9de7-a996e9c19882" />
</details>
<details><summary><strong>CNN + Residual Connections</strong></summary>
    <img width="1989" height="1589" alt="CNN + Residual Connections visualization" src="https://github.com/user-attachments/assets/6b832c0a-e784-4b92-b1bc-0c3a5a47ed91" />
</details>

<details><summary><strong>Transfer Learning (ResNet34)</strong></summary>
    <img width="1989" height="1589" alt="Transfer Learning (ResNet34) visualization" src="https://github.com/user-attachments/assets/cc1f7e36-e1e2-4735-8ae1-fbfa20b6decf" />

</details>


## 🔍 Grad-CAM Examples
> Grad-CAM was implemented to visualize regions in images that influenced the model's decision.

<strong>CNN from Scratch</strong>
<img width="989" height="343" alt="image" src="https://github.com/user-attachments/assets/dc665b70-0e7f-47ee-a957-46f372870d95" />



<strong>CNN + Residual Connections</strong>
<img width="989" height="343" alt="image" src="https://github.com/user-attachments/assets/c3276be0-9723-42a9-af02-625646ef1911" />


