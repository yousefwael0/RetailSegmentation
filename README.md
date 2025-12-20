# 🛒 Retail Buyer Segmentation System

An AI-powered customer segmentation system built with **Streamlit** and **Scikit-learn** that achieves **97.8% accuracy** in predicting customer segments based on purchasing behavior and demographics.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Information](#model-information)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)

---

## ✨ Features

- 🎯 **Single Customer Prediction**: Manual input form for individual customer segmentation
- 📁 **Batch Processing**: Upload CSV files to segment multiple customers at once
- 📊 **Interactive Visualizations**: Beautiful charts powered by Plotly
- 💡 **Marketing Recommendations**: Actionable insights for each customer segment
- 📈 **Confidence Scores**: Model prediction confidence for transparency
- 📥 **Export Results**: Download segmentation results as CSV
- 🎨 **Modern UI**: Clean, responsive interface built with Streamlit

### Customer Segments

1. **Budget-Conscious Families** (48% of customers)
   - Lower income households with focus on value
   - Average spend: $105

2. **Middle-Income Shoppers** (30% of customers)
   - Balanced buyers seeking quality and affordability
   - Average spend: $796

3. **Premium High-Spenders** (22% of customers)
   - High-income customers preferring premium products
   - Average spend: $1,429

---

## 📁 Project Structure

```
.
├── app.py                                  # Main Streamlit application
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
│
├── models/                                 # Trained models (generated from notebook)
│   ├── logistic_model.pkl                 # Logistic Regression classifier
│   ├── scaler.pkl                         # StandardScaler for features
│   └── feature_names.pkl                  # Feature configuration & cluster info
│
├── data/                                   # Generated cluster statistics
│   └── cluster_info.json                  # Cluster metadata (size, avg income, etc.)
│
├── data.csv                                # Original raw dataset
├── data_modeled_scaled.csv                # Processed & scaled dataset
│
├── utils/                                  # Utility modules
│   └── preprocessing.py                   # Feature engineering functions
│
├── RetailBuyerSegmentation.ipynb          # Jupyter notebook (model training)
├── Retail Buyer Segmentation Doc.pdf      # Project documentation
└── venv/                                   # Virtual environment (not tracked in git)
```

---

## 🔧 Requirements

### System Requirements

- **Python**: 3.8 or higher (tested on 3.12.4)
- **pip**: Latest version
- **Operating System**: Windows, macOS, or Linux

### Python Dependencies

All dependencies are listed in `requirements.txt`:

```txt
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
joblib==1.3.2
plotly==5.18.0
matplotlib==3.8.2
seaborn==0.13.0
```

---

## 🚀 Installation

### Step 1: Clone/Download the Repository

```bash
# If using git
git clone <repository-url>
cd RetailSegmentation

# Or download and extract the ZIP file
cd path/to/RetailSegmentation
```

### Step 2: Create Virtual Environment

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import sklearn; print('✅ All packages installed successfully!')"
```

---

## ⚡ Quick Start

### Option A: Models Already Exported (Ready to Run)

If the `models/` and `data/` folders already contain `.pkl` and `.json` files:

```bash
# 1. Activate virtual environment
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows

# 2. Run the app
streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

### Option B: Export Models First (Fresh Setup)

If `models/` is empty or you want to retrain:

1. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook RetailBuyerSegmentation.ipynb
   ```

2. **Run all cells** from top to bottom (Cell → Run All)

3. **Run the export cell** at the end of the notebook to generate:
   - `models/logistic_model.pkl`
   - `models/scaler.pkl`
   - `models/feature_names.pkl`
   - `data/cluster_info.json`

4. **Verify files were created:**
   ```bash
   ls -la models/
   ls -la data/
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

### Stopping the App

Press `Ctrl+C` in the terminal to stop the server.

---

## 📖 Usage Guide

### 🧑 Single Customer Prediction

1. **Select Input Method**
   - In the sidebar, choose **"🧑 Single Customer (Manual)"**

2. **Enter Customer Information**
   
   Fill in the three sections:
   
   **👤 Demographics:**
   - Birth Year (1940-2005)
   - Education Level (Basic, 2n Cycle, Graduation, Master, PhD)
   - Marital Status (Married, Together, Single, Divorced, Widow, Other)
   - Annual Income ($)
   - Number of Children
   - Number of Teenagers

   **🛍️ Shopping Behavior:**
   - Spending on: Wine, Fruits, Meat, Fish, Sweets, Gold Products

   **📱 Purchase Channels:**
   - Number of purchases: Web, Catalog, Store, Discount
   - Web visits last month
   - Campaign engagement (6 checkboxes)

3. **Get Prediction**
   - Click **"🔍 Predict Customer Segment"**

4. **Review Results**
   - Predicted segment with confidence score
   - Segment size and average spending
   - Probability distribution chart
   - Customer profile summary
   - Personalized marketing recommendations

### 📁 Batch Upload (CSV)

1. **Select Batch Mode**
   - In the sidebar, choose **"📁 Batch Upload (CSV)"**

2. **Prepare Your CSV**
   
   Required columns (use `data.csv` as a template):
   ```
   birth_year, education_level, marital_status, annual_income,
   num_children, num_teenagers, spend_wine, spend_fruits,
   spend_meat, spend_fish, spend_sweets, spend_gold,
   num_web_purchases, num_catalog_purchases, num_store_purchases,
   num_discount_purchases, web_visits_last_month,
   accepted_campaign_1, accepted_campaign_2, accepted_campaign_3,
   accepted_campaign_4, accepted_campaign_5, accepted_last_campaign
   ```

3. **Upload File**
   - Click "Choose a CSV file"
   - Select your file
   - Preview the first 10 rows

4. **Process Customers**
   - Click **"🚀 Process All Customers"**

5. **View Results**
   - Summary statistics (total customers, avg confidence, most common segment)
   - Segment distribution pie chart
   - Confidence distribution histogram
   - Detailed results table

6. **Download Results**
   - Click **"⬇️ Download Complete Results (CSV)"**
   - File includes: predicted cluster, cluster name, confidence score

### Example CSV Format

```csv
birth_year,education_level,marital_status,annual_income,num_children,num_teenagers,spend_wine,spend_fruits,spend_meat,spend_fish,spend_sweets,spend_gold,num_web_purchases,num_catalog_purchases,num_store_purchases,num_discount_purchases,web_visits_last_month,accepted_campaign_1,accepted_campaign_2,accepted_campaign_3,accepted_campaign_4,accepted_campaign_5,accepted_last_campaign
1980,Graduation,Married,55000,1,0,250,30,120,35,20,40,5,2,7,3,6,0,0,0,0,0,0
1965,PhD,Single,85000,0,0,800,50,400,80,30,100,8,5,10,1,4,1,0,0,1,0,1
```

---

## 🤖 Model Information

### Algorithm

- **Model**: Logistic Regression (Multinomial)
- **Training Accuracy**: 97.8%
- **Features**: 43 engineered features
- **Clusters**: 3 customer segments
- **Training Samples**: 2,240 customers

### Feature Engineering Pipeline

The model uses the following engineered features:

1. **Demographic Features**
   - `customer_age` = 2014 - birth_year
   - `education_encoded` (1-5 scale)
   - `marital_status_*` (one-hot encoded: Divorced, Married, Other, Single, Together, Widow)
   - `annual_income`

2. **Spending Features**
   - `total_spend` (sum of all spending categories)
   - `avg_spend` (mean spending)
   - `spend_wine_ratio` (wine spend / total spend)
   - `spend_meat_ratio` (meat spend / total spend)

3. **Purchase Channel Features**
   - `total_purchases` (web + catalog + store)
   - `web_purchase_ratio` (web / total)
   - `store_purchase_ratio` (store / total)

4. **Engagement Features**
   - `total_campaigns_accepted` (sum of all campaign responses)
   - `campaign_acceptance_rate` (accepted / total campaigns)

5. **Family Features**
   - `family_size` (children + teenagers)
   - `has_dependents` (binary: 0 or 1)

6. **Temporal Features**
   - `customer_tenure_days` (days since signup)

### Preprocessing Steps

1. **Feature Engineering** (`utils/preprocessing.py`)
   - Calculate derived features
   - Handle missing values
   - Encode categorical variables

2. **Scaling** (`models/scaler.pkl`)
   - StandardScaler (zero mean, unit variance)
   - Applied to all numeric features

3. **Prediction** (`models/logistic_model.pkl`)
   - Logistic Regression classifier
   - Returns cluster ID and probabilities

---

## 🐛 Troubleshooting

### ❌ Error: "Missing model files"

**Symptom:**
```
⚠️ Missing model files!
FileNotFoundError: [Errno 2] No such file or directory: 'models/logistic_model.pkl'
```

**Solution:**
```bash
# 1. Open the notebook
jupyter notebook RetailBuyerSegmentation.ipynb

# 2. Run all cells including the export cell at the end

# 3. Verify files exist
ls models/
# Should show: logistic_model.pkl  scaler.pkl  feature_names.pkl

ls data/
# Should show: cluster_info.json
```

---

### ❌ Error: "ModuleNotFoundError"

**Symptom:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# 1. Verify virtual environment is activated
which python  # Should point to venv/bin/python

# If not activated:
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 2. Reinstall requirements
pip install -r requirements.txt
```

---

### ❌ Error: "Port 8501 already in use"

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Option 1: Run on different port
streamlit run app.py --server.port 8502

# Option 2: Kill existing process (Mac/Linux)
lsof -ti:8501 | xargs kill -9

# Option 2: Kill existing process (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

### ⚠️ Warning: Scikit-learn version mismatch

**Symptom:**
```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.7.2 when using version 1.8.0
```

**Solution:**
```bash
# Match the version used for training
pip uninstall scikit-learn
pip install scikit-learn==1.3.2

# Then restart the app
streamlit run app.py
```

---

### ❌ Error: "KeyError: customer_tenure_days"

**Symptom:**
```
ValueError: Missing required columns: ['customer_tenure_days']
```

**Solution:**
This means the preprocessing code is outdated. The `customer_tenure_days` feature is automatically created with a default value. Ensure you have the latest `utils/preprocessing.py` file.

---

### 📊 Charts not displaying

**Symptom:**
Charts appear blank or don't load.

**Solution:**
```bash
# 1. Reinstall plotly
pip install --upgrade plotly

# 2. Clear Streamlit cache
streamlit cache clear

# 3. Restart the app
streamlit run app.py
```

---

## 🔄 Updating the Model

If you retrain the model with new data or parameters:

1. **Update the notebook** with new data or hyperparameters
2. **Run all cells** from top to bottom
3. **Run the export cell** to overwrite model files:
   ```python
   # The export cell creates:
   # - models/logistic_model.pkl
   # - models/scaler.pkl
   # - models/feature_names.pkl
   # - data/cluster_info.json
   ```
4. **Restart the Streamlit app**:
   ```bash
   # Stop app: Ctrl+C
   streamlit run app.py
   ```
5. The app will automatically load the new models (thanks to `@st.cache_resource`)

---

## 🚢 Deployment

### Local Deployment (Current Setup)

Already covered in [Quick Start](#quick-start)

### Streamlit Cloud (Free Hosting)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add retail segmentation app"
   git push origin main
   ```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file: `app.py`
   - Click "Deploy"

3. **Important:** Ensure model files are committed to git:
   ```bash
   git add models/*.pkl data/cluster_info.json
   ```

### Docker Deployment

**Create `Dockerfile`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
# Build image
docker build -t retail-segmentation .

# Run container
docker run -p 8501:8501 retail-segmentation

# Access at http://localhost:8501
```

---

## 📊 Performance Metrics

- **Accuracy**: 97.8%
- **Training Samples**: 2,240 customers
- **Features**: 43 engineered features
- **Clusters**: 3 (non-consecutive IDs: 0, 1, 3)
- **Prediction Time**: < 100ms per customer
- **Batch Processing**: ~1,000 customers/second

---

## 🎓 Academic Context

This project was developed as part of a retail analytics course demonstrating:
- Machine learning model development
- Feature engineering
- Web application deployment
- Business intelligence and customer segmentation

**Model Performance:** 97.8% accuracy on customer segmentation task.

---

## 📚 Additional Resources

- **Documentation**: See `Retail Buyer Segmentation Doc.pdf`
- **Notebook**: `RetailBuyerSegmentation.ipynb` for model training details
- **Dataset**: `data.csv` (original) and `data_modeled_scaled.csv` (processed)

---

## 📝 License

This project is licensed under the MIT License.

---

**Model Accuracy: 97.8% | 2,240 Training Samples | 3 Customer Segments**