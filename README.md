# Public Policy Insight Generator (PPIG)

## Project Overview
The **Public Policy Insight Generator (PPIG)** is an AI-powered web application designed to help policymakers and governance teams quickly analyze citizen feedback on public policies. The application leverages **Aspect-Based Sentiment Analysis (ABSA)** using **Gemini AI** to extract sentiments related to specific policy components such as **Cost, Safety, Accessibility, and Effectiveness**.

PPIG transforms unstructured public feedback into a concise, **actionable 3-point policy report**, enabling rapid, data-driven decision-making for effective governance. The application is built using **Streamlit** for an intuitive user interface and is deployed for public access.

---

## Objective
- **Sentiment Understanding:** Analyze public opinion on specific policy aspects
- **Aspect-Level Insights:** Identify strengths and weaknesses of policy components
- **Actionable Reporting:** Generate a clear 3-point policy recommendation report
- **Rapid Governance:** Support faster, data-driven policy decisions
- **Citizen-Centric Analysis:** Amplify citizen voices using AI-driven insights

---

## Tools & Technologies Used

### Languages & Frameworks
- **Python** – Core programming language
- **Streamlit** – Interactive web application framework

### AI & Natural Language Processing
- **Gemini AI (Google)** – Aspect-Based Sentiment Analysis (ABSA)
- **Prompt Engineering** – Structured prompts for insight generation

### Data Handling
- **Pandas** – Data manipulation and preprocessing

### Deployment
- **Streamlit Cloud** – Public deployment platform
- **GitHub** – Version control and collaboration

---

## Workflow Summary

### 1. Input Collection
- Users input citizen feedback text related to a public policy
- Feedback may include opinions on multiple policy components

---

### 2. Aspect-Based Sentiment Analysis (ABSA)
- Gemini AI analyzes feedback at the **aspect level**
- Key aspects identified include:
  - Cost
  - Safety
  - Accessibility
  - Implementation
  - Public Impact
- Each aspect is classified with sentiment:
  - Positive
  - Negative
  - Neutral

---

### 3. Insight Generation
- Sentiment results are aggregated across aspects
- Key trends and concerns are identified
- AI generates a **3-point actionable policy insight report**, focusing on:
  - What is working well
  - What needs improvement
  - Recommended policy actions

---

### 4. Streamlit Application
- Clean, user-friendly interface
- Text input for citizen feedback
- One-click analysis and report generation
- Clear visualization of insights and recommendations

---

## Deployment
- Source code hosted on **GitHub**
- Application deployed using **Streamlit Cloud**
- Automatic updates on new GitHub commits
- Public URL available for real-time access

---

## Live Deployment

| Platform        | Link |
|-----------------|------|
| Streamlit Cloud | https://public-policy-insight-generator-ppig.streamlit.app/ |

---

## How to Run Locally

### 1. Clone the Repository
git clone https://github.com/student-cod/Public-Policy-Insight-Generator-PPIG-.git
cd Public-Policy-Insight-Generator-PPIG-

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Streamlit App
streamlit run app.py
