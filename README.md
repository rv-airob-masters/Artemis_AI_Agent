# ARTEMIS : : Advanced Result & Test Evaluation Medical Information System – AI-Powered Medical Analyser 🧠⚕️

**Artemis** is an intelligent medical analyser tool developed as part of the **University of Hertfordshire's extracurricular AI agent creation activity**. It processes lab reports (PDF/image), extracts test data, compares values against standard ranges, and offers insightful explanations using AI.

---

## 🚀 Features

- **Text Extraction via Azure**: Extracts medical test text from reports (PDFs/images) using **Microsoft Azure Document Intelligence API** for high-accuracy OCR.
- **Data Structuring with LLMs**: Extracted text is parsed and structured using large language models to identify key test components like glucose, WBC count, etc.
- **Reference Range Comparison**: Automatically checks if the report includes ranges; otherwise falls back to a **SQLite database** built from NHS data categorised by **gender and age**.
- **Medical Analysis Agent (Mistral AI)**: Flags abnormal values and provides **AI-driven reasoning**, probable causes, and possible remedies – with a **medical disclaimer**.
- **Interactive Chat Agent (Gemini AI)**: Allows users to ask questions about their results, causes, improvements, and general medical insights in natural language.
- **Modular Architecture**: The Medical Analysis Agent and Chat Agent are implemented in **separate modules** to ensure clean separation and flexible extensibility.
- **User-Friendly UI**: Built with Streamlit for seamless interaction.

---

## 🧩 Requirements

- Python 3.8+
- All dependencies are listed in `requirements.txt`

---

## 🔐 API Keys & Environment Setup

The application uses external APIs (Azure, Anthropic, Mistral, Gemini). Store your API keys in a `.env` file at the root of the project.

### Example `.env`:

```
LANGSMITH_API_KEY="Your Langsmith key"
GEMINI_API_KEY="Your Google gemini Key"
AZURE_DOCU_KEY="Your Azure Document Intelligence Key"
AZURE_DOCU_URL="Your Azure Document Intelligence URL endpoint"
MISTRAL_API_KEY="Your Mistral key"
```

---

## 🛠️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/rv-airob-masters/artemis-labinsight.git
cd artemis-labinsight
```

### 2. Set up the environment

```bash
pip install -r requirements.txt
```

### 3. Add API keys

Create a `.env` file and paste your API keys as shown above.

### 4. Run the app

```bash
streamlit run src_files/TestResultAnalyser_agents.py
```

---

## 📂 Project Structure 

```
artemis-labinsight/
├── db/
│   └── med_reference_ranges.db
├── src_files/
│   ├── blood_test_agent.py
│   ├── medical_analyzer.py
│   ├── ocr_library.py
│   └── TestResultAnalyser_agents.py
├── README.md
├── requirements.txt
├── .env
```

---

## 📐 Architecture

```
                    +--------------------------+
                    |  User Uploads PDF/Image  |
                    +------------+-------------+
                                 |
                                 v
              +--------------------------------------+
              | Azure Document Intelligence API (OCR)|
              +--------------------------------------+
                                 |
                                 v
                  +-------------------------------+
                  |  LLM Parser (Test Data Extract)|
                  +-------------------------------+
                                 |
                                 v
     +------------------------- Reference Range Engine -------------------------+
     | Checks if report includes ranges. If not, queries SQLite database        |
     | (Built from NHS data: categorised by age/gender)                         |
     +--------------------------------------------------------------------------+
                                 |
                                 v
               +-------------------------------------------+
               |     🔍 Medical Analysis Agent (Mistral AI)|
               | - Detects abnormalities                  |
               | - Suggests causes & remedies             |
               | - Adds medical disclaimer                |
               +-------------------------------------------+

                                 |
                      +---------+----------+
                      |                    |
                      v                    v
         +------------------+     +----------------------------+
         | Structured Output|     | 💬 Chat Agent (Gemini AI)  |
         | & Flags          |     | - Q&A on results           |
         |                  |     | - Lifestyle guidance       |
         +------------------+     +----------------------------+

                                 |
                                 v
                     +-----------------------------+
                     | Streamlit Frontend UI       |
                     | - Upload / Preview Reports  |
                     | - Results & Explanations    |
                     | - Interactive Chat          |
                     +-----------------------------+
```

---

## 📽️ Demo Video

[Watch ARTEMIS in action 🎬](https://tinyurl.com/krm3ybnp)

---

## 📘 License

MIT License. See [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements

- Developed as part of the **University of Hertfordshire AI Agent Creation Activity** with technical support from **Dr Mohammed Bahja**
- OCR via **Microsoft Azure Document Intelligence API**
- Reasoning powered by **Mistral AI**
- Chat module powered by **Google Gemini**
- Reference range data parsed from [NHS Haematology Ranges](https://www.gloshospitals.nhs.uk/our-services/services-we-offer/pathology/haematology/haematology-reference-ranges/)
