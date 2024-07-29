from langchain_community.document_loaders import DirectoryLoader
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from typing import Literal
from app.enums.constants import Config
from app.Prompts.prompt_templates import Prompts
from app.models.langgraph_models import GraphState, GraphConfig, GradeHallucinations, GradeAnswer

class DocumentWorkflow:
    def __init__(self):
        self.loader = DirectoryLoader('Documents', glob="**/*.txt")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily_search_tool = TavilySearchResults(max_results=1)

        # Load documents and initialize embedding and retriever
        docs = self.loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        embedding_function = OpenAIEmbeddings()
        self.db = FAISS.from_documents(docs, embedding_function)

    def document_search(self, state: GraphState):
        """Retrieve documents."""
        print("---RETRIEVE---")
        question = convert_to_messages(state["messages"])[-1].content
        documents = self.db.similarity_search(question, k=1)
        return {"documents": documents, "question": question, "web_fallback": True}

    def generate(self, state: GraphState):
        """Generate answer."""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        retries = state["retries"] if state.get("retries") is not None else -1
        rag_chain = Prompts.RAG_PROMPT | self.llm | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"retries": retries + 1, "candidate_answer": generation}

    def transform_query(self, state: GraphState):
        """Transform the query to produce a better question."""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        query_rewriter = Prompts.QUERY_REWRITER_PROMPT | self.llm | StrOutputParser()
        better_question = query_rewriter.invoke({"question": question})
        return {"question": better_question}

    def web_search(self, state: GraphState):
        """Perform web search."""
        print("---RUNNING WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        search_results = self.tavily_search_tool.invoke(question)
        search_content = "\n".join([d["content"] for d in search_results])
        documents.append(Document(page_content=search_content, metadata={"source": "websearch"}))
        return {"documents": documents, "web_fallback": False}

    def grade_generation_v_documents_and_question(self, state: GraphState, config) -> Literal["generate", "transform_query", "web_search", "finalize_response"]:
        """Determine whether the generation is grounded in the document and answers the question."""
        question = state["question"]
        documents = state["documents"]
        generation = state["candidate_answer"]
        web_fallback = state["web_fallback"]
        retries = state["retries"] if state.get("retries") is not None else -1
        max_retries = config.get("configurable", {}).get("max_retries", Config.MAX_RETRIES.value)

        if not web_fallback:
            return "finalize_response"

        print("---CHECK HALLUCINATIONS---")
        hallucination_grader = Prompts.HALLUCINATION_GRADER_PROMPT | self.llm.with_structured_output(GradeHallucinations)
        hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )

        if hallucination_grade.binary_score == "no":
            return "generate" if retries < max_retries else "web_search"

        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

        print("---GRADE GENERATION vs QUESTION---")
        answer_grader = Prompts.ANSWER_GRADER_PROMPT | self.llm.with_structured_output(GradeAnswer)
        answer_grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "finalize_response"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "transform_query" if retries < max_retries else "web_search"

    def finalize_response(self, state: GraphState):
        """Finalize the response."""
        print("---FINALIZING THE RESPONSE---")
        return {"messages": [AIMessage(content=state["candidate_answer"])]}

    def create_workflow(self):
        """Build and compile the workflow graph."""
        workflow = StateGraph(GraphState, config_schema=GraphConfig)

        # Define nodes
        workflow.add_node("document_search", self.document_search)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("finalize_response", self.finalize_response)

        # Define edges
        workflow.set_entry_point("document_search")
        workflow.add_edge("document_search", "generate")
        workflow.add_edge("transform_query", "document_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("finalize_response", END)

        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question
        )

        return workflow.compile()
