"""
title: Google Manifold Pipeline
author: justinh-rahb
date: 2024-05-29
version: 1.0
license: MIT
description: A pipeline for generating text using the Google Gemini API.
dependencies: requests, google-generativeai
environment_variables: GOOGLEAI_API_KEY
"""

import os
import google.generativeai as genai
from typing import List, Union, Generator
from pydantic import BaseModel


class Pipeline:
    def __init__(self):
        self.type = "manifold"
        self.id = "google"
        self.name = "google/"
        
        class Valves(BaseModel):
            GOOGLEAI_API_KEY: str
            
        self.valves = Valves(GOOGLEAI_API_KEY=os.getenv("GOOGLEAI_API_KEY"))
        genai.configure(api_key=self.valves.GOOGLEAI_API_KEY)
        
    def get_google_models(self) -> List[dict]:
        # List available Google Gemini models
        return [
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-pro", "name": "Gemini Pro"},
            {"id": "gemini-pro-vision", "name": "Gemini Pro Vision"},
            # Add other models as they are released
        ]

    async def on_startup(self):
        print(f"Starting up: {__name__}")

    async def on_shutdown(self):
        print(f"Shutting down: {__name__}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator]:
        if messages is None:
            messages = []
        messages.append({"role": "user", "content": user_message})
        
        params = self.translate_parameters(body)
        # Check if streaming is requested
        if body.get("stream", False):
            return self.stream_response(model_id, messages, params)
        else:
            return self.get_completion(model_id, messages, params)

    def translate_parameters(self, body: dict) -> dict:
        return {
            "max_output_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 0.9),
            "stop_sequences": body.get("stop", [])
        }

    def stream_response(
        self, model_id: str, messages: List[dict], params: dict
    ) -> Generator:
        model = genai.GenerativeModel.init(model_id=model_id)
        response = model.generate_text(
            prompt=messages[-1]['content'],
            stream=True,
            **params
        )

        # Gemini doesn't use chunks with types. Yield text directly.
        for chunk in response:
            yield chunk.text 

    def get_completion(
        self, model_id: str, messages: List[dict], params: dict
    ) -> str:
        model = genai.GenerativeModel.init(model_id=model_id)
        response = model.generate_text(
            prompt=messages[-1]['content'],
            **params
        )
        return response.text
