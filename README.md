# EDA-Driven-Analysis-of-Social-Media-Behavior-Mental-Health-Risk-Patterns.
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Social%20Media%20%26%20Mental%20Health&fontSize=40&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=EDA-Driven%20Risk%20Analysis%20%7C%20Behavioral%20Segmentation%20%7C%20Sentiment%20Intelligence&descAlignY=55&descSize=14" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Viz-4C8CBF?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![ReportLab](https://img.shields.io/badge/ReportLab-PDF-E74C3C?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com)

<br/>

[![Dataset](https://img.shields.io/badge/Dataset-SMMH.csv%20%7C%20481%20responses-success?style=flat-square)](https://www.kaggle.com)
[![Clusters](https://img.shields.io/badge/Segments-4%20Behavioral%20Personas-blueviolet?style=flat-square)]()
[![CV F1](https://img.shields.io/badge/GBM%20CV%20F1-0.91+-orange?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)]()

</div>

---

## 📌 Overview

> **Can social media usage patterns predict mental health risk?**
> This project uses a full end-to-end EDA pipeline — from raw survey responses to ML-powered behavioral personas — to answer that question with statistical rigor and visual depth.

Working with **481 survey responses** across 21 variables, this analysis uncovers hidden patterns between platform usage, social comparison behavior, and five key mental health dimensions: **depression, anxiety, sleep disruption, validation-seeking, and concentration loss**.

---

## 🗂️ Table of Contents

- [Dataset](#-dataset)
- [Pipeline Overview](#-pipeline-overview)
- [Key Findings](#-key-findings)
- [Behavioral Segments](#-behavioral-segments)
- [Sentiment Analysis](#-sentiment-analysis)
- [ML Models](#-ml-models)
- [Visual Highlights](#-visual-highlights)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 📊 Dataset

| Field | Detail |
|---|---|
| **Source** | Kaggle — Social Media & Mental Health Survey (`smmh.csv`) |
| **Rows** | 481 respondents (477 after cleaning) |
| **Columns** | 21 (demographics + 11 Likert-scale MH items) |
| **Scale** | 1–5 Likert (1 = Never / Not at all, 5 = Very Often / Extremely) |
| **Platforms Covered** | Instagram, TikTok, YouTube, Facebook, Twitter, Reddit, Pinterest, Snapchat |

**Mental health features measured:**
`purposeless` · `distracted_busy` · `restless` · `easily_distracted` · `worries` · `diff_concentrating` · `social_comparison` · `validation` · `depression` · `interest_fluctuation` · `sleep`

---

## 🔁 Pipeline Overview

```
Raw CSV (smmh.csv)
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  STEP 1 → Problem Definition                             │
│  STEP 2 → Data Import & Examination                      │
│  STEP 3 → Missing Value Treatment                        │
│  STEP 4 → Exploratory Characteristic Analysis            │
│  STEP 5 → Feature Engineering (MH Score, Risk Levels)    │
│  STEP 6 → Relationship Visualisation (Heatmaps, Scatter) │
│  STEP 7 → Outlier Detection (IQR + Isolation Forest)     │
│  STEP 8 → PCA + KMeans + Random Forest Importance        │
│  STEP 9 → Sentiment Analysis (comparison_feel column)    │
│  STEP 10 → Behavioral Segmentation + GBM Classifier      │
│  STEP 11 → PDF Report Generation (ReportLab)             │
└──────────────────────────────────────────────────────────┘
       │
       ▼
  Behavioral_Segmentation_Report.pdf
```

---

## 🔍 Key Findings

### 📈 Depression vs. Daily Usage

```
Avg Depression Score (1–5)
─────────────────────────────────────────────────────────
< 1 hr   │████░░░░░░░░░░░░░░░░░░░░  2.1
1–2 hrs  │██████░░░░░░░░░░░░░░░░░░  2.5
2–3 hrs  │████████░░░░░░░░░░░░░░░░  2.8
3–4 hrs  │███████████░░░░░░░░░░░░░  3.2
4–5 hrs  │█████████████░░░░░░░░░░░  3.5
> 5 hrs  │████████████████░░░░░░░░  3.9
         └────────────────────────────────
          1.0    2.0    3.0    4.0    5.0
```

> **More usage = higher depression.** Users spending 5+ hours/day score ~86% higher in depression than those using under 1 hour.

---

### 🧠 Top MH Risk Predictors (Pearson |r| with MH Score)

```
social_comparison    │████████████████████████████░  0.72 ★
worries              │███████████████████████░░░░░░  0.68
depression           │██████████████████████░░░░░░░  0.65
validation           │████████████████████░░░░░░░░░  0.61
sleep                │██████████████████░░░░░░░░░░░  0.57
easily_distracted    │████████████████░░░░░░░░░░░░░  0.53
interest_fluctuation │██████████░░░░░░░░░░░░░░░░░░░  0.41
                     └─────────────────────────────────────
                      0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7
```

> 🥇 **Social comparison is the single strongest predictor of mental health risk** — stronger even than reported depression scores.

---

### 👥 Risk by Occupation

```
Occupation          │  Avg MH Score  │  Risk Profile
────────────────────┼────────────────┼──────────────────────
University Student  │  ●●●●○  3.4   │  🔴 HIGH
Salaried Employee   │  ●●●○○  2.9   │  🟡 MODERATE
School Student      │  ●●●○○  2.8   │  🟡 MODERATE
Self-Employed       │  ●●○○○  2.4   │  🟢 LOW-MODERATE
Homemaker           │  ●●○○○  2.2   │  🟢 LOW
Retired             │  ●○○○○  1.8   │  🟢 LOW
```

---

### 🎂 Risk by Age Group

```
Age Group   Avg MH Score
──────────────────────────────────────────
< 18     │  ██████████████░  2.8
18–24    │  ████████████████████  3.4  ← PEAK
25–30    │  ██████████████████░  3.2
31–40    │  ████████████░░░░░░  2.6
40+      │  ████████░░░░░░░░░  2.1
         └──────────────────────────────
           1.0    2.0    3.0    4.0
```

---

## 🎭 Behavioral Segments

Using **K-Means (k=4)** on 11 standardised features, four statistically distinct user personas emerged (validated by Silhouette score):

<table>
<thead>
<tr>
<th>Segment</th>
<th>% of Users</th>
<th>Avg MH Score</th>
<th>Avg Daily Usage</th>
<th>Dominant Feature</th>
<th>Top Demographic</th>
</tr>
</thead>
<tbody>
<tr>
<td>🟢 <b>Balanced Users</b></td>
<td>~28%</td>
<td>1.9 / 5.0</td>
<td>1.5 hrs/day</td>
<td>Low across all</td>
<td>Older, employed</td>
</tr>
<tr>
<td>🔵 <b>Anxious Overthinkers</b></td>
<td>~24%</td>
<td>2.7 / 5.0</td>
<td>3.0 hrs/day</td>
<td>Worries + Distraction</td>
<td>School students</td>
</tr>
<tr>
<td>🟡 <b>Social Comparers</b></td>
<td>~25%</td>
<td>3.1 / 5.0</td>
<td>3.5 hrs/day</td>
<td>Social Comparison</td>
<td>University students</td>
</tr>
<tr>
<td>🔴 <b>High-Risk Addicts</b></td>
<td>~23%</td>
<td>4.1 / 5.0</td>
<td>5.2 hrs/day</td>
<td>Depression + Sleep</td>
<td>Young adults 18–24</td>
</tr>
</tbody>
</table>

### Radar Profile Summary

```
        Depression
            5 ┐
            4 ┤     🔴
            3 ┤  🟡   🔵
            2 ┤🟢
            1 ┘
  Sleep ◄──────────────► Worries
            1 ┐
            2 ┤🟢
            3 ┤  🔵 🟡
            4 ┤     🔴
            5 ┘
       Social Comparison
```

> The **🔴 High-Risk Addicts** cluster scores highest on every dimension.
> The **🟢 Balanced Users** cluster sits consistently below average on all features — they represent healthy engagement benchmarks.

---

## 💬 Sentiment Analysis

The `comparison_feel` column (how users feel after comparing themselves to others) was mapped to polarity scores and analysed against all MH metrics.

### Sentiment Distribution

```
Very Negative  │████████████░░░░░░░░░░░░░░  22%
Negative       │████████████████░░░░░░░░░░  31%
Neutral        │████████░░░░░░░░░░░░░░░░░░  18%
Positive       │██████████░░░░░░░░░░░░░░░░  19%
Very Positive  │████░░░░░░░░░░░░░░░░░░░░░░  10%
               └───────────────────────────────
Overall mood: 😞 Mostly Negative  |  Avg polarity: –0.18
```

### Key Statistical Tests

| Test | Feature | Result | Significant? |
|---|---|---|---|
| Kruskal-Wallis | social_comparison | H=48.61, p<0.0001 | ✅ YES |
| Kruskal-Wallis | depression | H=41.3, p<0.0001 | ✅ YES |
| Kruskal-Wallis | mh_score | H=39.8, p<0.0001 | ✅ YES |
| Spearman ρ | polarity vs mh_score | ρ = –0.44, p<0.0001 | ✅ YES |
| Chi-Square | Sentiment ↔ Risk Level | χ²=38.2, p<0.0001 | ✅ YES |

> **Negative sentiment after social comparisons is strongly and significantly associated with higher mental health risk scores across every tested dimension.**

### Sentiment by Platform (Avg Polarity)

```
Pinterest    │  ██████████████  +0.12  😊
YouTube      │  ████████████░░  +0.05  😐
Reddit       │  ███████████░░░  –0.03  😐
Facebook     │  █████████░░░░░  –0.08  😕
Twitter/X    │  ███████░░░░░░░  –0.14  😞
TikTok       │  █████░░░░░░░░░  –0.19  😞
Instagram    │  ████░░░░░░░░░░  –0.22  😞  ← Most negative
             └──────────────────────────────────
              –0.25     0.0     +0.25
```

---

## 🤖 ML Models

### Random Forest — Feature Importance (MH Risk Prediction)

```
depression           ████████████████████████  0.18
social_comparison    ███████████████████████░  0.16
worries              ██████████████████████░░  0.15
sleep                ████████████████░░░░░░░░  0.12
validation           ███████████████░░░░░░░░░  0.11
easily_distracted    █████████████░░░░░░░░░░░  0.09
interest_fluctuation ████████████░░░░░░░░░░░░  0.08
diff_concentrating   █████████░░░░░░░░░░░░░░░  0.06
distracted_busy      ███████░░░░░░░░░░░░░░░░░  0.03
restless             █████░░░░░░░░░░░░░░░░░░░  0.02
purposeless          ████░░░░░░░░░░░░░░░░░░░░  0.01
```

### Logistic Regression — Sentiment Prediction

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.72 | 0.68 | 0.70 |
| Neutral | 0.54 | 0.59 | 0.56 |
| Positive | 0.69 | 0.71 | 0.70 |
| **Macro Avg** | **0.65** | **0.66** | **0.65** |

> 5-Fold CV F1 (macro): **0.648 ± 0.031**

### GBM Segment Classifier

| Segment | AUC Score |
|---|---|
| 🟢 Balanced Users | 0.97 |
| 🔴 High-Risk Addicts | 0.96 |
| 🟡 Social Comparers | 0.89 |
| 🔵 Anxious Overthinkers | 0.87 |

> 5-Fold CV F1 (macro): **0.91 ± 0.02** — clusters are statistically well-separated.

---

## 🖼️ Visual Highlights

The notebook generates **18 publication-quality charts** across 9 analysis stages:

| # | Figure | Description |
|---|---|---|
| 1 | **Missing Value Treatment** | Before/after bar charts |
| 2 | **Data Characteristics** | 6-panel: age, gender, occupation, platforms, usage, depression |
| 3 | **Feature Transformations** | MH score distribution, risk levels, scaling comparison |
| 4 | **Relationship Heatmap** | 12×12 Pearson correlation matrix |
| 5 | **Usage vs Depression** | Bar chart with trend |
| 6 | **Outlier Detection** | 6-panel scatter with IQR bounds |
| 7 | **PCA + Cluster Plot** | 2D projection with feature importance |
| 8 | **Sentiment Overview** | 4-panel: distribution, polarity, gender, occupation heatmap |
| 9 | **Violin Deep-Dive** | Sentiment vs 6 MH metrics |
| 10 | **Statistical Tests** | Kruskal-Wallis + Spearman + Chi-Square printed output |
| 11 | **Platform Sentiment** | Polarity bar, stacked %, MH boxplot |
| 12 | **Heatmap + Radar** | Sentiment × Risk Level + polar chart |
| 13 | **Logistic Regression** | Coefficient heatmap + confusion matrix |
| 14 | **Sentiment × Usage** | Stacked bar, polarity line, scatter with colormap |
| 15 | **Elbow + Silhouette** | Optimal k selection |
| 16 | **Segmentation Overview** | PCA, pie, heatmap, MH bar |
| 17 | **Radar Profiles** | 4-panel polar charts per segment |
| 18 | **GBM + ROC Curves** | Feature importance + one-vs-rest ROC |

---

## 📁 Project Structure

```
📦 social-media-mental-health-eda
 ┣ 📜 eda_driven_analysis_of_social_media_behavior___mental_health_risk_patterns.py
 ┣ 📊 smmh.csv                          ← Input dataset (upload manually)
 ┣ 📄 Behavioral_Segmentation_Report.pdf ← Auto-generated PDF report
 ┣ 📁 outputs/
 ┃  ┣ 🖼️ seg_elbow.png
 ┃  ┣ 🖼️ seg_main.png
 ┃  ┣ 🖼️ seg_radar.png
 ┃  ┣ 🖼️ seg_demographics.png
 ┃  ┣ 🖼️ seg_classifier.png
 ┃  ┗ 🖼️ seg_distribution.png
 ┗ 📖 README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy reportlab
```

### Running in Google Colab (Recommended)

1. Open the `.py` file in Google Colab
2. Upload `smmh.csv` when prompted (available on [Kaggle](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health))
3. Run all cells sequentially — charts render inline, PDF saves to `/content/`

### Running Locally

```python
# Replace the Colab file upload block with:
df = pd.read_csv("smmh.csv")
```

Then run the script end-to-end. Output PNG files and the PDF report will be saved to the configured `OUT` directory.

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Tools |
|---|---|
| **Data Wrangling** | `pandas` · `numpy` |
| **Visualisation** | `matplotlib` · `seaborn` |
| **Statistics** | `scipy.stats` (Kruskal-Wallis, Spearman, Chi-Square) |
| **Machine Learning** | `scikit-learn` (KMeans, PCA, RF, GBM, Logistic Regression, IsolationForest) |
| **Report Generation** | `reportlab` (multi-page PDF with tables, figures, styled sections) |
| **Environment** | Google Colab / Python 3.10+ |

</div>

---

## 💡 Recommendations

Based on the analysis findings:

- 🔴 **High-Risk Addicts** should be the primary target for digital wellness interventions — they show the steepest scores on depression and sleep disruption simultaneously
- 🟡 **Social Comparers** benefit most from platform-level awareness campaigns, particularly for the 18–24 university student demographic
- 📱 **Platform designers** should consider usage-limit nudges for users logging 5+ hours/day
- 🧘 **Anxious Overthinkers** respond well to mindfulness-based digital detox programs
- 🟢 **Balanced Users** provide a healthy engagement benchmark — studying their habits can inform positive design patterns

---

## 📃 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Built with Python · Scikit-Learn · Seaborn · ReportLab**

*If this project helped you, consider giving it a ⭐*

</div>
