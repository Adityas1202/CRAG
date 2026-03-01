import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.utils import get_retriever, get_web_search_tool

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key)

def retrieve(state):
    #print("---NODE: RETRIEVE---")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    #print("---NODE: GRADE---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"grade": "fail", "documents": documents, "question": question}

    # The "Judge" prompt
    grade_prompt = f"""Evaluate if the following document is relevant to the user question.
    Document: {documents}
    Question: {question}
    Respond with only one word: 'relevant' or 'irrelevant'."""
    
    result = llm.invoke(grade_prompt).content.strip().lower()
    
    if "irrelevant" in result:
      #  print("---GRADE: IRRELEVANT (Triggering Web Search)---")
        return {"grade": "fail", "documents": documents, "question": question}
    else:
     #   print("---GRADE: RELEVANT---")
        return {"grade": "success", "documents": documents, "question": question}

def transform_query(state):
    #print("---NODE: TRANSFORM QUERY---")
    question = state["question"]
    
    # Strict prompt to prevent the AI from "explaining"
    better_question_prompt = f"""You are a search query optimizer. 
    Convert the user's input into a high-quality search engine query.
    Output ONLY the optimized query text. Do not include any explanations.
    
    User Input: {question}
    Optimized Query:"""
    
    response = llm.invoke(better_question_prompt)
    optimized_q = response.content.strip()
    
   # print(f"---TRANSFORMED TO: {optimized_q}---")
    return {"question": optimized_q}

def web_search(state):
    #print("---NODE: WEB SEARCH---")
    question = state["question"]
    search_tool = get_web_search_tool()
    
    # Invoke the tool
    results = search_tool.invoke({"query": question})
    
    # FIX: Check if results is a string or a list
    if isinstance(results, str):
        # If it's a string, use it directly as the content
        web_content = results
    else:
        # If it's a list of dicts, join the content fields
        web_content = "\n".join([d.get("content", "") for d in results if isinstance(d, dict)])
    
    return {"documents": web_content, "question": question}

def generate(state):
   # print("---NODE: GENERATE---")
    # FIX: Use 'question' from the state dictionary
    question = state["question"] 
    documents = state["documents"]
    
    prompt = f"""You are a precise Fact-Checker. 
    Use the provided Web Search Context to answer the question accurately.
    Today is March 2, 2026. If the context contains match results from yesterday (March 1, 2026), prioritize those.
    
    Context: {documents}
    Question: {question} 
    Answer:"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content}