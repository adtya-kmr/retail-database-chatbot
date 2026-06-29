
import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from few_shots import few_shots

# =============================================================================
# Environment
# =============================================================================

load_dotenv()

# Load API key from Streamlit Cloud if available
try: 
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] # Streamlit Cloud except Exception: pass
except Exception: 
    pass

# =============================================================================
# Models
# =============================================================================

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# =============================================================================
# Database
# =============================================================================


db = SQLDatabase.from_uri(
    "sqlite:///database/retail.db"
)


# =============================================================================
# Few-shot Retrieval
# =============================================================================

example_texts = [
    f"""Question:
{ex['input']}

SQL Pattern:
{ex['sql_cmd']}

SQL Result:
{ex['sql_result']}

Expected Answer:
{ex['answer']}
"""
    for ex in few_shots
]

few_shot_vectorstore = Chroma.from_texts(
    texts=example_texts,
    embedding=embedding_model,
    metadatas=few_shots,
)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=few_shot_vectorstore,
    k=2,
)

# =============================================================================
# Prompts
# =============================================================================

sql_prompt = """You are an expert MySQL developer.

Generate ONLY a valid MySQL query that answers the user's question.

You will receive:
- Database schema
- Similar example question-query pairs

Use the examples as guidance. Reuse their SQL patterns where appropriate,
but adapt them to the current question.

Rules:
- Return ONLY SQL.
- No explanations.
- No markdown.
- No SQLResult.
- No Answer.
- Use only existing tables and columns.
- Do not invent values.
"""

sql_suffix = """
Table Information:
{table_info}

Question:
{input}

SQL Query:
"""

answer_prompt = """You are a retail assistant.

Answer the user's question using ONLY the SQL result.

Rules:
- Never output SQL.
- Never explain SQL.
- Keep the response concise and conversational.
- If no records are found, say so politely.
"""

answer_suffix = """
Question:
{input}

SQL Result:
{sql_result}

Answer:
"""

# =============================================================================
# Prompt Templates
# =============================================================================

sql_example_prompt = PromptTemplate(
    input_variables=["input", "sql_cmd"],
    template="Question:\n{input}\n\nSQL Query:\n{sql_cmd}",
)

answer_example_prompt = PromptTemplate(
    input_variables=["input", "sql_result", "answer"],
    template="Question:\n{input}\n\nSQL Result:\n{sql_result}\n\nAnswer:\n{answer}",
)

sql_few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=sql_example_prompt,
    prefix=sql_prompt,
    suffix=sql_suffix,
    input_variables=["input", "table_info"],
)

answer_few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=answer_example_prompt,
    prefix=answer_prompt,
    suffix=answer_suffix,
    input_variables=["input", "sql_result"],
)

# =============================================================================
# Public API
# =============================================================================

def get_few_shot_db_chain():
    """
    Build and return the SQL generation and answer generation chains.

    Returns:
        tuple:
            (sql_chain, answer_chain)
    """

    sql_chain = (
        sql_few_shot_prompt
        | gemini_llm
        | StrOutputParser()
    )

    answer_chain = (
        answer_few_shot_prompt
        | gemini_llm
        | StrOutputParser()
    )

    return sql_chain, answer_chain
