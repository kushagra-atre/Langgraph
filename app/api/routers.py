from fastapi import APIRouter, HTTPException
from app.models.chat_request_model import QueryRequest
from app.controller.workflow_controller import DocumentWorkflow

router = APIRouter()
doc_workflow = DocumentWorkflow()

@router.post("/process")
async def process_query(request: QueryRequest):
    try:
        graph = doc_workflow.create_workflow()
        inputs = {"messages": [("human", request.content)]}
        results = graph.invoke(inputs)
        ai_messages_content = [message.content for message in results["messages"] if message.type == "ai"]
        return {"results": ai_messages_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
