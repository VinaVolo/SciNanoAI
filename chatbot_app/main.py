import os
from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse
from chatbot import ChatBot

app = FastAPI()

chatbot_instance = ChatBot(llm_model="openai/gpt-4o-mini")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        user_message = request.message
        response_message = chatbot_instance.generate_response(user_message)
        conversation_history = chatbot_instance.conversation_history
        return ChatResponse(reply=response_message, conversation_history=conversation_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@app.post("/clear_history")
def clear_history():
    chatbot_instance.conversation_history = []
    return {"message": "История очищена"}
