# 🌐 Multilingual Sentiment Analysis Project (Portuguese & English)

## 🎯 Project Objective
The goal of this project is to create an **automated sentiment analysis tool** capable of identifying whether a text expresses a **positive, negative, or neutral opinion**, also classifying the emotional trend (optimistic or pessimistic) and the confidence level of the analysis.

The solution processes texts in **Portuguese** and **English**, allowing the user to manually input new sentences for evaluation or use predefined examples.

---

## 💡 Solution Description
The project leverages **pre-trained natural language processing (NLP) models** from the **Hugging Face Transformers** library, offering high performance and accuracy for sentiment analysis tasks without the need to train a model from scratch.

Two main versions are implemented:

### 🇧🇷 Portuguese Version
Uses the model **`nlptown/bert-base-multilingual-uncased-sentiment`**, which supports multiple languages including Portuguese.  
It classifies sentences on a **1 to 5 stars scale**, where:
- ⭐ 1 star → negative sentiment  
- ⭐ 3 stars → neutral sentiment  
- ⭐ 5 stars → positive sentiment  

### 🇬🇧 English Version
Uses the model **`distilbert-base-uncased-finetuned-sst-2-english`**, developed specifically for English.  
It categorizes sentences as:
- **POSITIVE** → Optimistic  
- **NEGATIVE** → Pessimistic  

Results are displayed in a **tabular format (DataFrame)** including:
| Analyzed Text | Classification | Confidence | Trend |
|---------------|----------------|-----------|-------|
| Loved the service, it was excellent! | 5 stars / POSITIVE | 0.75 | Optimistic |
| The service was slow and I was unsatisfied. | 1 star / NEGATIVE | 0.62 | Pessimistic |
| It was okay, nothing special. | 1 star / NEGATIVE | 0.49 | Pessimistic |

---

## 🧩 Features
- Automatic sentiment detection for **Portuguese and English**.  
- Option for users to **add custom sentences** at runtime.  
- Results displayed in **tabular format** (DataFrame).  
- Confidence level for predictions.  
- Additional classification of **emotional trend** (optimistic/pessimistic).  

---

## ⚙️ Technologies Used
- **Python 3.11**  
- **Hugging Face Transformers**  
- **PyTorch**  
- **Pandas**  
- **Scikit-learn** (optional for additional analysis)  
- **ipykernel** (for Jupyter Notebook integration)  

---

## 💬 Why Not Use Azure AI
Initially, the project considered using **Azure Cognitive Services – Text Analytics**, which also provides sentiment analysis and language detection.  
However, this approach was discarded for the following reasons:

1. 💰 **Variable costs per API request:**  
   Azure charges per number of analyzed texts, which can result in unpredictable costs during testing or continuous use.

2. ⚡ **Limitations in local development environments:**  
   API integration requires a constant internet connection, limiting offline testing or notebook execution.

3. 🔐 **Preference for technological independence:**  
   Using open-source models ensures the project is **fully autonomous and free**, running on any machine, including offline.

4. 🧠 **Full control over models and data:**  
   Local models allow customization, fine-tuning, and transparency in predictions, ensuring reproducibility.

---

## 🚀 Practical Applications
- Analysis of **customer feedback** in Portuguese and English.  
- Monitoring **product reviews** and **social media comments**.  
- Support for **satisfaction surveys** and **brand reputation monitoring**.  
- Educational tool to demonstrate **NLP applied to business and data**.

---

## 🧩 Possible Extensions
- Automatic **language detection** to select the correct model.  
- Interactive interface using **Streamlit** or **Gradio**.  
- Integration with **Power BI** for sentiment dashboards.  
- Deployment via **Flask or FastAPI API** for corporate use.

---

## ✅ Conclusion
This project demonstrates the power of **open-source NLP models** applied to sentiment analysis, providing a **cost-free, efficient, and customizable alternative** to paid cloud services.  
It enables **accurate sentiment analysis in multiple languages**, broadening the use cases for companies, research, and educational applications.

---

## ⚡ Setup Instructions
```bash
# Create and activate the environment
conda create -n nlp_env python=3.11 -y
conda activate nlp_env

# Install PyTorch (CPU-only)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install NLP packages
pip install transformers scikit-learn scipy pandas ipykernel

