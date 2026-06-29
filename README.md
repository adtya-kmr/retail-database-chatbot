# 👕 Retail Q&A Tool

A Generative AI-powered application that converts natural language questions into SQL queries, executes them on a SQLite database, and returns conversational answers.

## Features

* Natural language to SQL using **Gemini 3.1 Flash Lite**
* Semantic few-shot example retrieval with **ChromaDB**
* SQL execution on **SQLite**
* Conversational answer generation
* Interactive Streamlit web interface

## Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini
* ChromaDB
* SQLite
* SQLAlchemy

## Application

<p align="center">
  <img src="assets/ui.png" width="850">
</p>

## Example Questions

* Which product earns the most after discounts?
* Which brand has the highest inventory value?
* How much cash is locked in Levi inventory?
* Which supplier has the highest inventory value?

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Retail-QA-Tool.git
cd Retail-QA-Tool
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

```text
GOOGLE_API_KEY=your_google_api_key
```

### 4. Run the application

```bash
streamlit run app.py
```

## Project Structure

```text
Retail-QA-Tool/
│
├── assets/
│   └── ui.png
│   
├── database/
│   ├── retail.db
│   └── retail_dataset.sql
│
├── app.py
├── langchain_helper.py
├── few_shots.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

