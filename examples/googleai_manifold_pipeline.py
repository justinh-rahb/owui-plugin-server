"""
title: Google Manifold Pipeline
author: justinh-rahb
date: 2024-05-29
version: 1.0
license: MIT
description: A pipeline for generating text using the Google Gemini API.
requirements: requests, google-generativeai
environment_variables: GOOGLEAI_API_KEY
"""

import os
import google.generativeai as genai
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests

class Pipeline:
    def __init__(self):
        self.type = "manifold"
        self.id = "google"
        self.name = "google/"
        
        class Valves(BaseModel):
            GOOGLEAI_API_KEY: str
            
        self.valves = Valves(GOOGLEAI_API_KEY=os.getenv("GOOGLEAI_API_KEY"))
        genai.configure(api_key=self.valves.GOOGLEAI_API_KEY)

    def get_google_models(self):
        # This could fetch models dynamically from Google in the future
        return [
            {"id": "gemini-1.5-flash", "name": "gemini-1.5-flash"},
            {"id": "gemini-1.5-pro", "name": "gemini-1.5-pro"},
            {"id": "gemini-pro", "name": "gemini-pro"},
            {"id": "gemini-pro-vision", "name": "gemini-pro-vision"},
            # Add other Google models as they become available
        ]

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        genai.configure(api_key=self.valves.GOOGLEAI_API_KEY)
        pass

    def pipelines(self) -> List[dict]:
        return self.get_google_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            if body.get("stream", False):
                return self.stream_response(model_id, messages, body)
            else:
                return self.get_completion(model_id, messages, body)
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, model_id: str, messages: List[dict], body: dict) -> Generator:
        params = self.translate_parameters(body)
        model = genai.GenerativeModel(model_id=model_id)  # Corrected instantiation
        response = model.generate_text(
            prompt=messages[-1]['content'],  
            stream=True,
            **params
        )

        for chunk in response:
            yield chunk.text  # Assume chunks come with a 'text' attribute

    def get_completion(self, model_id: str, messages: List[dict], body: dict) -> str:
        params = self.translate_parameters(body)
        model = genai.GenerativeModel(model_id=model_id)  # Corrected instantiation
        response = model.generate_text(
            prompt=messages[-1]['content'],
            **params
        )
        return response.text

    def translate_parameters(self, body: dict) -> dict:
        return {
            "max_output_tokens": body.get("max_tokens", 4096),
            "temperature": body.get("temperature", 0.8),
            "top_k": body.get("top_k", 40),
            "top_p": body.get("top_p", 0.9),
            "stop_sequences": body.get("stop", [])
        }
