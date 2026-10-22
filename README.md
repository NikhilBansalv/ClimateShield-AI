# 🌍 ClimateShield AI

### Personalized Climate Risk Assessment and Adaptation Planner

ClimateShield AI is a Python-based climate analysis project designed to study historical temperature and precipitation patterns across Indian regions, identify regional climate-exposure indicators, and establish a foundation for personalized climate adaptation planning.

The current version analyzes historical climate data from 1991 to 2025 across 35 Indian sub-national regions using data from the World Bank Climate Change Knowledge Portal (CCKP).

> **Project status:** Historical data analysis and preliminary climate-exposure scoring are implemented. The interactive dashboard and adaptation-planning features are under development and refinement.

## 🎯 Objectives

- Collect and analyze historical temperature and precipitation data.
- Identify regional differences in temperature exposure and rainfall variability.
- Analyze historical temperature trends and regional warming rates.
- Develop a transparent, preliminary climate-exposure scoring framework.
- Visualize regional climate indicators and assessment results.
- Establish a foundation for region-specific climate adaptation recommendations.

## ✨ Current Features

- **Data preprocessing:** Cleans, reshapes and integrates historical temperature and precipitation datasets.
- **Exploratory data analysis:** Examines regional temperature, annual precipitation and rainfall variability.
- **Temperature trend analysis:** Estimates regional linear temperature trends over the study period.
- **Feature engineering:** Calculates temperature anomalies and rainfall deviations relative to regional historical averages.
- **Preliminary climate scoring:** Combines normalized temperature indicators and rainfall variability into a comparative score.
- **Visualizations:** Generates charts highlighting regional climate patterns and differences.
- **Interactive dashboard:** A Streamlit interface is being developed to explore climate indicators and regional scores.
- **Adaptation planning:** Rule-based suggestions are being explored as an initial step toward region-specific guidance.

## 📊 Dataset

| Property               | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| Data source            | World Bank Climate Change Knowledge Portal (CCKP)        |
| Historical dataset     | CRU climate data                                         |
| Spatial coverage       | 35 Indian sub-national regions                           |
| Study period           | 1991–2025                                                |
| Temperature variable   | `tasmax` — average daily maximum surface air temperature |
| Precipitation variable | `pr` — precipitation                                     |
| Temporal aggregation   | Annual                                                   |
| Final analysis dataset | 1,225 region-year records                                |

The project uses historical climate indicators to study regional patterns. The results should not be interpreted as direct measurements of individual vulnerability or future disaster probability.

## 🧮 Preliminary ClimateShield Scoring Methodology

The current scoring framework uses min–max normalization to scale the selected indicators from 0 to 100 across the analyzed regions.

### Heat Exposure Score

- 50% — Normalized mean maximum temperature
- 30% — Normalized linear temperature trend
- 20% — Normalized recent temperature change

The recent temperature change compares the 2021–2025 mean with the 1991–2020 baseline.

### Rainfall Variability Stress Score

- 100% — Normalized rainfall coefficient of variation

### Overall ClimateShield Score

- 70% — Heat Exposure Score
- 30% — Rainfall Variability Stress Score

The current implementation assigns preliminary Low, Moderate and High categories based on the resulting score.

**Important limitation:** These are project-defined relative indicators, not validated disaster-risk categories. The score depends on the analyzed regions and selected indicators and has not been validated against observed disaster impacts.

## 🛠️ Technology Stack

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- SciPy
- Streamlit and Plotly for the interactive dashboard

## 📁 Project Structure

```text
ClimateShield-AI/
├── data/
│   ├── gw_tsmax.csv
│   └── gw_pr.csv
├── outputs/
│   ├── ClimateShield_DA1_Cleaned_Dataset_1991_2025.csv
│   ├── ClimateShield_Risk_Assessment.csv
│   ├── regional_summary.csv
│   ├── temperature_trends.csv
│   └── graphs/
├── main.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The exact files in `outputs/` depend on which analysis steps have been run. The `app.py` dashboard requires the cleaned dataset and risk-assessment CSV produced by `main.py`.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ClimateShield-AI.git
cd ClimateShield-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the historical analysis

```bash
python main.py
```

This processes the source datasets, generates analysis outputs and saves the regional risk assessment.

### 5. Launch the interactive dashboard

```bash
python -m streamlit run app.py
```

The dashboard can then be accessed at the local URL displayed in the terminal, typically `http://localhost:8501`.

## 🔍 Key Research Questions

- Which regions have the highest historical average maximum temperatures?
- Which regions exhibit the strongest estimated temperature warming trends?
- How does annual precipitation vary across regions?
- Which regions exhibit the greatest rainfall variability?
- How do the selected indicators influence the preliminary ClimateShield score?

## ⚠️ Limitations

- The current analysis uses two climate variables.
- Annual data cannot fully capture short-duration extreme weather events.
- Population exposure, socioeconomic vulnerability and infrastructure are not included.
- The preliminary score has not been validated against observed disaster or impact data.
- Historical trends are not future climate forecasts.
- Adaptation suggestions are preliminary and do not replace official local climate or emergency guidance.

## 🔮 Future Scope

- Incorporate additional climate indicators and extreme-event measures.
- Explore future climate projections and scenarios.
- Refine and evaluate the climate-exposure scoring methodology.
- Improve the interactive dashboard and regional comparison tools.
- Develop more context-aware adaptation recommendations.
- Evaluate additional predictive methods if suitable data and validation targets are available.

## 📚 Data Source and Acknowledgment

Historical climate data are obtained from the [World Bank Climate Change Knowledge Portal](https://climateknowledgeportal.worldbank.org/download-data).

Please consult the source's documentation and applicable data-use terms when redistributing the datasets. Cite the CCKP and the relevant underlying dataset in academic reports and derivative work.

## 👨‍💻 Project Information

**Project:** ClimateShield AI — Personalized Climate Risk Assessment and Adaptation Planner

**Domain:** Climate Change, Data Science and Climate-Exposure Assessment

**Implementation:** Python-based data analysis with a developing Streamlit application

---

_ClimateShield AI aims to make historical climate information easier to explore and interpret, supporting the development of more informed, region-specific climate adaptation planning._
