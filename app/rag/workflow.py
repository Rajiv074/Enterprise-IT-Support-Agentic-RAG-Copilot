import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
settings = get_settings()
_llm = None
_web_search = None
def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        # _llm = ChatOpenAI(
        #     model=settings.openai_model,
        #     temperature=0,
        #     api_key=settings.openai_api_key,
        # )
        _llm = ChatGroq(
            model_name=settings.groq_model,
            temperature=0,
            # max_output_tokens=512,
            api_key=settings.groq_api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


from langchain_core.prompts import ChatPromptTemplate

def route_question(state: AgentState):
    question = state["question"]

    # 1. Define the router prompt
    router_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a router for an Agentic RAG assistant.

            Route to "kb" for questions about:
            - Agentic RAG and LangGraph workflows
            - Retrieval grading and query rewriting
            - RAG architecture and retriever tools
            - Web fallback in RAG
            - Other informational or technical questions

            Route to "direct" only for:
            - Greetings
            - Thanks
            - Simple conversation

            Return your response as valid JSON.
            The JSON must contain exactly one field:
            "route", with value "kb" or "direct".

            Example: {{"route": "kb"}}
            """
        ),
        ("human", "{question}")
    ])

    # 2. Create the structured LLM chain
    router_chain = (
        router_prompt
        | llm().with_structured_output(
            RouteDecision,
            method="json_mode"
        )
    )

    # 3. Invoke the router
    decision = router_chain.invoke({
        "question": question
    })

    # 4. Return updated state
    return {
        "current_query": question,
        "source_used": decision.route,
        "trace": add_trace(state, f"Router → {decision.route.upper()}")
    }

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"

def retrieve_kb(state: AgentState):
    query = state["current_query"]
    kb_docs = get_retriever().invoke(query)
    print(f"[KB Retrieval] query: '{query}'")
    print(f"[KB Retrieval] Retrieved {len(kb_docs)} chunks.")
    return {
        "kb_docs": kb_docs,
        "trace": add_trace(state, f"Private KB retrieval → {len(kb_docs)} chunks")
    }


def grade_kb_evidence(state: AgentState):
    question = state["question"]

    # 1. Prepare the KB evidence
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in state["kb_docs"]
    )

    # 2. Define the grader prompt
    grader_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an evidence grader for an Agentic RAG assistant.

            Evaluate whether the private KB evidence can answer
            the user's question.

            - Return "good" if the evidence is relevant and sufficient
              to answer the question.
            - Return "weak" if the evidence is irrelevant or incomplete
              and cannot answer the question.

            Return your response as valid JSON.
            Example:
            {{"grade": "good"}}

            The grade must be either "good" or "weak".
            Do not include additional text or markdown.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Private KB evidence:
            {context}
            """
        )
    ])

    # 3. Configure structured output
    kb_grader_llm = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode"
    )

    # 4. Create the grader chain
    kb_grader_chain = grader_prompt | kb_grader_llm

    # 5. Invoke the grader
    grade = kb_grader_chain.invoke({
        "question": question,
        "context": context
    })

    print("[KB Grader]", grade.grade)

    # 6. Return the updated state
    return {
        "kb_grade": grade.grade,
        "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}")
    }

def decide_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }


def grade_web_evidence(state: AgentState):
    question  = state["question"]
    web_results = state["web_results"]
    
    web_grader_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an evidence grader..

            Evaluate whether the web search evidence can answer
            the user's question.

            - Return "good" if the evidence is relevant and sufficient
              to answer the question.
            - Return "weak" if the evidence is irrelevant or incomplete
              and cannot answer the question.

            Return your response as valid JSON.
            Example:
            {{"grade": "good"}}

            The grade must be either "good" or "weak".
            Do not include additional text or markdown.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Web search evidence::
            {web_results}
            """
        )
    ])
     # 3. Configure structured output
    web_grader_llm = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode"
    )

    # 4. Create the grader chain
    web_grader_chain = web_grader_prompt | web_grader_llm

    # 5. Invoke the grader
    grade = web_grader_chain.invoke({
        "question": question,
        "web_results": web_results
    })

    print("[KB Grader]", grade.grade)

    # 6. Return the updated state
    return {
        "web_grade": grade.grade,
        "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")
    }
    
    # conditional decision making after web grading


def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "answer_insufficient"



def rewrite_query(state: AgentState):
    question = state["question"]
    retry_count = state["retry_count"] + 1

    # 1. Define the query rewriting prompt
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a query rewriting assistant for an Agentic RAG system.

            Rewrite the user's question to improve retrieval from the
            knowledge base and web search.

            Rules:
            - Preserve the original intent.
            - Make the query specific and search-friendly.
            - Include relevant technical terms where appropriate.
            - Do not answer the question.
            - Return only the rewritten query.
            """
        ),
        (
            "human",
            """
            Original question:
            {question}
            """
        )
    ])

    # 2. Create the LLM chain
    rewrite_chain = rewrite_prompt | llm()

    # 3. Invoke the chain
    response = rewrite_chain.invoke({
        "question": question
    })

    rewritten = response.content.strip()

    print("[Rewriter]", rewritten)

    # 4. Return the updated state
    return {
        "current_query": rewritten,
        "retry_count": retry_count,
        "trace": add_trace(state, f"Query rewrite → {rewritten}")
    }
    
def generate_from_kb(state: AgentState):
    question = state["question"]
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    kb_answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a technical instructor.

        Answer using ONLY the private KB context.

        Rules:
        - Beginner-friendly explanation.
        - Do not invent unsupported details.
        - Mention that the answer is based on the private KB.
        - Include source type: Private KB.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Private KB evidence:
            {context}
            """
        )
    ])
    # 2. Create the LLM chain
    kb_answer_chain = kb_answer_prompt | llm()

    # 3. Invoke the chain
    response = kb_answer_chain.invoke({
        "question": question,
        "context": context
    })

    answer = response.content.strip()
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})

    print("[private_KB_answer]", answer)
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}


def generate_from_web(state: AgentState):
    question = state["question"]
    context = state["web_results"]
    
    web_answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a technical instructor.

            The private KB was insufficient, so web search was used.

            Answer using ONLY the web search context.

            Rules:
            - Beginner-friendly explanation.
            - Do not invent unsupported details.
            - Mention that the answer is based on Tavily web search.
            - Include source type: Web Search.
            - If URLs are present in context, include the most useful URLs.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Web search context:
            {context}
            """
        )
    ])
    # 2. Create the LLM chain
    web_answer_chain = web_answer_prompt | llm()

    # 3. Invoke the chain
    response = web_answer_chain.invoke({
        "question": question,
        "context": context
    })

    answer = response.content.strip()

    print("[web answer]", answer)

    # 4. Return the updated state
    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(state, "Direct response → no retrieval")
    }
    

def direct_answer(state: AgentState):
    question = state["question"]

    direct_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Respond briefly and naturally. 
        """
    ),(
        "human",
        """
        Question:
        {question}
        """
    )])
    # 2. Create the LLM chain
    web_answer_chain = direct_answer_prompt | llm()

    # 3. Invoke the chain
    response = web_answer_chain.invoke({
        "question": question,
    })

    answer = response.content.strip()

    print("[direct answer]", answer)

    # 4. Return the updated state
    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(state, "Direct response → no retrieval")
    }
   

def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }
    

def build_graph():
    agentic_rag_graph_workflow = StateGraph(AgentState)

    # 1. Register all nodes first
    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb_evidence": grade_kb_evidence,
        "search_web": search_web,
        "grade_web_evidence": grade_web_evidence,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "answer_insufficient": answer_insufficient,
    }

    for name, fn in nodes.items():
        agentic_rag_graph_workflow.add_node(name, fn)

    # 2. Connect START to the router
    agentic_rag_graph_workflow.add_edge(
        START, "route_question"
    )

    # 3. Route the user's question
    agentic_rag_graph_workflow.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    # 4. Retrieve documents and grade evidence
    agentic_rag_graph_workflow.add_edge(
        "retrieve_kb", "grade_kb_evidence"
    )

    agentic_rag_graph_workflow.add_conditional_edges(
        "grade_kb_evidence",
        decide_after_kb_grade,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    # 5. Search the web and grade web evidence
    agentic_rag_graph_workflow.add_edge(
        "search_web", "grade_web_evidence"
    )

    agentic_rag_graph_workflow.add_conditional_edges(
        "grade_web_evidence",
        decide_after_web_grade,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )

    # 6. Retry retrieval using the rewritten query
    agentic_rag_graph_workflow.add_edge(
        "rewrite_query", "retrieve_kb"
    )

    # 7. Connect terminal nodes to END
    agentic_rag_graph_workflow.add_edge(
        "generate_from_kb", END
    )
    agentic_rag_graph_workflow.add_edge(
        "generate_from_web", END
    )
    agentic_rag_graph_workflow.add_edge(
        "direct_answer", END
    )
    agentic_rag_graph_workflow.add_edge(
        "answer_insufficient", END
    )

    # 8. Compile only after all nodes and edges are registered
    return agentic_rag_graph_workflow.compile()


agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)


    