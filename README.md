# MedCheck: AI-Assisted Medication Interaction Checker

Educational medical AI prototype for a short workshop. Enter two or more medication names to check possible drug interactions and view basic medicine information from CSV datasets.

**This tool does not train or fine-tune any machine-learning model.** It uses a pretrained Sentence Transformer only to match medication names.

---

## Features

- Multi-medication input (one per line or comma-separated)
- Automatic CSV column detection and flexible name mapping
- Medication name matching with:
  - Exact normalized match
  - RapidFuzz spelling fallback
  - MiniLM semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
  - Combined scoring with a configurable confidence threshold (default **0.75**)
- Pairwise interaction checks (order-independent)
- Dataset severity labels or keyword-inferred severity (clearly marked)
- Overall risk levels: Green / Yellow / Red
- Gemini Flash rewrites of interaction descriptions into short plain-English summaries
- Medicine details: generic/salt, uses, side effects, substitutes
- Graceful fallback if the embedding model cannot download
- Optional Plotly severity chart

---

## Project structure

```
medcheck/
├── app.py                      # Streamlit UI
├── requirements.txt
├── README.md
├── data/
│   ├── drug_interactions.csv   # Replace with your dataset
│   └── medicine_details.csv    # Replace with your dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # load_datasets, detect_columns, build_unique_drug_list
│   ├── preprocessing.py        # normalize_drug_name, input parsing
│   ├── medication_matcher.py   # embeddings + match_medication_name
│   ├── interaction_checker.py  # pairs, find_drug_interaction, infer_severity
│   ├── risk_scoring.py         # calculate_overall_risk
│   └── medicine_details.py     # get_medicine_details
└── assets/                     # Optional images / logo
```

---

## Dataset setup

Place your CSV files here:

- `data/drug_interactions.csv`
- `data/medicine_details.csv`

If the files are missing, the app shows Streamlit uploaders in the sidebar.

### Interaction dataset — expected concepts

| Logical field | Example column names |
|---------------|----------------------|
| Drug 1 | `Drug 1`, `drug_1`, `Drug A` |
| Drug 2 | `Drug 2`, `drug_2`, `Drug B` |
| Interaction | `Interaction`, `Description` |
| Severity (optional) | `Severity`, `Level`, `Risk` |

### Medicine details dataset — expected concepts

| Logical field | Example column names |
|---------------|----------------------|
| Name | `Medicine Name`, `Medicine_Name`, `name` |
| Generic / salt | `Generic Name`, `Salt Composition` |
| Uses | `Uses`, `Indications` |
| Side effects | `Side Effects`, `side_effects` |
| Substitutes | `Substitutes`, `Substitute0`, `Substitute1`, … |

Sample CSVs are included so workshop demos work immediately. **Replace them with your own datasets** when ready — column names will be auto-mapped when possible.

---

## Installation

```bash
cd medcheck
pip install -r requirements.txt
```

The first run may download `sentence-transformers/all-MiniLM-L6-v2` (internet required once). If download fails, MedCheck continues with exact + RapidFuzz matching only.

### Gemini Flash setup

MedCheck can use Gemini Flash to rewrite technical interaction sentences for non-medical users.

Set your API key in either environment variables:

```bash
set GEMINI_API_KEY=your_api_key_here
```

Or in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL = "gemini-2.0-flash"
```

`GEMINI_MODEL` is optional. If no API key is configured, the app falls back to the original dataset wording.

---

## How to run

```bash
cd medcheck
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## Pretrained model (no training)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

- Used **only** to create embeddings for medication-name matching
- Cosine similarity (scikit-learn) ranks the closest dataset names
- **Do not** call `fit()`, fine-tune, or train a new model
- Model and embeddings are cached with Streamlit (`@st.cache_resource` / `@st.cache_data`)

---

## Matching logic

1. Exact normalized match → confidence 100%
2. Otherwise RapidFuzz + MiniLM (when available)
3. Best combined / compared score wins
4. Below threshold → not auto-accepted; user sees a warning and can accept, reject, or pick another name

Default threshold: **0.75** (adjustable in the sidebar).

---

## Overall risk levels

| Level | When |
|-------|------|
| 🟢 No Known Interaction Found | No pairs found in the interaction dataset |
| 🟡 Moderate Risk | At least one moderate interaction; no major/severe |
| 🔴 High Risk | At least one major / severe / contraindicated interaction |

One major interaction is enough for High Risk. Highest severity always takes priority over total score.

**Never treat “no known interaction” as “safe.”** The UI states that absence from the dataset does not guarantee safety.

---

## Sample test cases

| Test | Input | Expected |
|------|-------|----------|
| 1 | Warfarin, Aspirin, Ibuprofen | High Risk (major bleeding interactions in sample data) |
| 2 | Aspirin, Ibuprofen | Moderate Risk (sample labels this pair Moderate) |
| 3 | Metformin, Amlodipine | No known interaction in the sample dataset |
| 4 | Warfarine, Ibuprophen | Suggests Warfarin and Ibuprofen via fuzzy/semantic match |

These cases are also available as buttons inside the app.

---

## Limitations

- Coverage is limited to whatever appears in your CSVs
- Fuzzy/semantic matching can still pick the wrong drug
- Keyword-inferred severity is **not** clinically validated
- Not evaluated for clinical decision support
- Educational demonstration only

---

## Medical disclaimer

This application is an educational prototype. It identifies possible medication interactions based only on the available datasets. It does not replace advice from a doctor, pharmacist, or other qualified healthcare professional. Do not start, stop, substitute, or change any medication based only on this tool.

Substitute medicines are shown for informational purposes only and must be approved by a qualified professional.
