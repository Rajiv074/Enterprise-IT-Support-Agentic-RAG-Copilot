from typing import List , Literal,Optional
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

#Crete Pydantic models for structured decision making
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(description="The route decision, either 'kb' for knowledge base or 'direct' for direct answer. 'kb' means the agent should use the knowledge base to find relevant information, 'direct' means the agent should provide a direct answer without using the knowledge base.")
    
class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(description="The grade of the evidence, either 'good' or 'weak' good means the evidence is relevant and reliable, weak means it is not.")

#Create a TypedDict for the agent state it will be used to store the state of the agent during the decision making process
class AgentState(TypedDict):
    question:str #your user query will come here
    current_query:str # rewrite your current query
    kb_docs: List[Document] # retrieve docs
    web_results: str # web result post tavily search
    kb_grade:str # knowledge base grade
    web_grade:str # web search result is good or weak
    answer:str # final answer
    source_used:str # source which it took to generate the answer
    retry_count:int # max retry 
    trace: List[str]
    citations: List[dict] # resulted output as well as source of result
