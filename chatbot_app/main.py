from fastapi import FastAPI, HTTPException

try:  # Support running via `uvicorn main:app` from inside chatbot_app directory.
    from chatbot_app.chatbot import ChatbotSettings, create_chatbot_service
    from chatbot_app.models import ChatRequest, ChatResponse
except ModuleNotFoundError:  # pragma: no cover - only for local CLI usage
    import importlib
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    chatbot_module = importlib.import_module("chatbot_app.chatbot")
    models_module = importlib.import_module("chatbot_app.models")
    ChatbotSettings = chatbot_module.ChatbotSettings
    create_chatbot_service = chatbot_module.create_chatbot_service
    ChatRequest = models_module.ChatRequest
    ChatResponse = models_module.ChatResponse

app = FastAPI()

settings = ChatbotSettings.from_env()
chatbot_service = create_chatbot_service(settings)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Main entrypoint for text generation requests.
    """

    try:
        response_message = chatbot_service.generate_response(request.message)
        return ChatResponse(
            reply=response_message,
            conversation_history=chatbot_service.conversation_history,
        )
    except Exception as exc:  # pragma: no cover - FastAPI handles the response.
        raise HTTPException(status_code=500, detail=f"Ошибка: {exc}") from exc


@app.post("/clear_history")
def clear_history():
    """
    Clears the persisted conversation history.
    """

    chatbot_service.clear_history()
    return {"message": "История очищена"}
