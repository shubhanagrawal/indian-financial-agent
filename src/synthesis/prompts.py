from langchain.prompts import PromptTemplate

# --- Prompts for General Q&A (Map-Reduce) ---

GENERAL_MAP_PROMPT = """
You are a helpful AI assistant. Your task is to extract key information from the following document chunk that is relevant to the user's question.
Read the document chunk carefully and identify any facts, figures, or statements that directly address the question.

CONTEXT:
{context}

QUESTION:
{question}

Extract the relevant information concisely:
"""
GENERAL_MAP_PROMPT_TEMPLATE = PromptTemplate(
    template=GENERAL_MAP_PROMPT, 
    input_variables=["context", "question"]
)

GENERAL_REDUCE_PROMPT = """
You are a helpful AI assistant. You have been provided with several summaries extracted from different documents.
Your task is to synthesize these summaries into a single, coherent, and comprehensive answer to the user's question.
Do not repeat information. Ensure the final answer is well-structured and easy to read.

SUMMARIES:
{context}

QUESTION:
{question}

Provide a final, synthesized answer based on the summaries:
"""
GENERAL_REDUCE_PROMPT_TEMPLATE = PromptTemplate(
    template=GENERAL_REDUCE_PROMPT, 
    input_variables=["context", "question"]
)


# --- Prompts for SWOT Analysis (Map-Reduce) ---

SWOT_MAP_PROMPT = """
You are a financial analyst. Based on the document chunk below, identify any points that could be considered a Strength, Weakness, Opportunity, or Threat for the company.
List only the points you find. If no relevant points are found in this chunk, say "No relevant SWOT points found."

CONTEXT:
{context}

QUESTION:
{question}

Identify potential SWOT points:
"""
SWOT_MAP_PROMPT_TEMPLATE = PromptTemplate(
    template=SWOT_MAP_PROMPT, 
    input_variables=["context", "question"]
)

# --- FINAL, MORE ADVANCED REDUCE PROMPT ---
SWOT_REDUCE_PROMPT = """
You are a senior financial analyst. You have been provided with a list of potential Strengths, Weaknesses, Opportunities, and Threats extracted from various documents.
Your final task is to synthesize these points into a professional, well-structured SWOT analysis.

Follow these instructions carefully:
1.  Create four distinct sections: ### Strengths, ### Weaknesses, ### Opportunities, and ### Threats.
2.  For each section, list the relevant points as clear and concise bullet points.
3.  **Crucially, you must IGNORE** any points that state "No relevant SWOT points found". Do not include this phrase in your final output.
4.  Do not repeat points. If you see similar points, consolidate them into a single, more general point.
5.  Base your final analysis *only* on the list of points provided below. Do not add any outside information.

LIST OF POINTS:
{context}

Synthesize these points into a final SWOT analysis, following the structured format described above:
"""
SWOT_REDUCE_PROMPT_TEMPLATE = PromptTemplate(
    template=SWOT_REDUCE_PROMPT, 
    input_variables=["context", "question"]
)
