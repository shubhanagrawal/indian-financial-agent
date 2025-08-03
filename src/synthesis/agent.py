from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, MapReduceDocumentsChain, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import traceback

# Import our new, specialized prompts
from prompts import (
    GENERAL_MAP_PROMPT_TEMPLATE, GENERAL_REDUCE_PROMPT_TEMPLATE,
    SWOT_MAP_PROMPT_TEMPLATE, SWOT_REDUCE_PROMPT_TEMPLATE
)

def create_qa_agent(company_name: str, use_swot_prompt: bool = False):
    """
    Initializes and returns a RetrievalQA agent using the advanced Map-Reduce strategy
    with a more powerful LLM and specialized prompts for higher quality answers.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        vector_store_path = project_root / f'data/vector_store/{company_name}'

        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 5})

        # --- UPGRADE: Using a more powerful model ---
        model_name = "google/flan-t5-large" # This model is better at reasoning
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer,
            max_length=512, # Max length of the generated answer
            temperature=0.1, top_p=0.95, repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        # --- Using specialized prompts for map and reduce steps ---
        if use_swot_prompt:
            print("(Using specialized SWOT prompts with Map-Reduce)")
            map_prompt = SWOT_MAP_PROMPT_TEMPLATE
            reduce_prompt = SWOT_REDUCE_PROMPT_TEMPLATE
        else:
            print("(Using specialized Q&A prompts with Map-Reduce)")
            map_prompt = GENERAL_MAP_PROMPT_TEMPLATE
            reduce_prompt = GENERAL_REDUCE_PROMPT_TEMPLATE

        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="context"
        )
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=combine_documents_chain,
            document_variable_name="context",
            return_intermediate_steps=False,
        )

        qa_chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=map_reduce_chain,
            return_source_documents=True
        )
        
        return qa_chain

    except Exception as e:
        print(f"[ERROR] An error occurred during agent initialization: {e}")
        traceback.print_exc()
        return None

if __name__ == '__main__':
    target_company = "RELIANCE_INDUSTRIES"
    
    print("--- Initializing Financial Synthesis Agent (Final Version) ---")
    print("This may take a moment, especially the first time, as a larger model will be downloaded.")
    
    agent = None

    print("\n--- Agent is Ready ---")
    print("Ask any question. To perform a SWOT analysis, include words like 'SWOT', 'strengths', etc.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break

        swot_keywords = ['swot', 'strengths', 'weaknesses', 'opportunities', 'threats']
        is_swot_query = any(keyword in query.lower() for keyword in swot_keywords)

        try:
            print("\nThinking... (This will be slower due to the larger model and advanced strategy)")
            agent = create_qa_agent(target_company, use_swot_prompt=is_swot_query)
            
            if agent:
                result = agent.invoke({"query": query})

                print("\n--- Result ---")
                print(result['result'])
                
                print("\n--- Sources ---")
                unique_sources = {source.metadata.get('source', 'Unknown source') for source in result['source_documents']}
                for source in sorted(list(unique_sources)):
                    print(f"- {Path(source).name}")
            else:
                print("Could not create the agent. Please check for errors above.")

        except Exception as e:
            print(f"\n[ERROR] An error occurred while processing your query: {e}")
