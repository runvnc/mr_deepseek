from mindroot.services import service_class
from mindroot.protocols import LLM
import os
import base64
from io import BytesIO
from openai import AsyncOpenAI
import json
from typing import AsyncIterator, Any

client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def concat_text_lists(message):
    """Concatenate text lists into a single string"""
    out_str = ""
    if isinstance(message['content'], str):
        return message
    else:
        for item in message['content']:
            if isinstance(item, str):
                out_str += item + "\n"
            else:
                out_str += item['text'] + "\n"
    message.update({'content': out_str})
    return message


@service_class(LLM)
class DeepSeekLLM(LLM):
    """DeepSeek LLM implementation using OpenAI-compatible API."""
    
    async def stream_chat(
        self,
        model: str,
        messages: list = None,
        context: Any = None,
        num_ctx: int = 200000,
        temperature: float = 0.0,
        max_tokens: int = 5000,
        num_gpu_layers: int = 0
    ) -> AsyncIterator[str]:
        try:
            reasoning = False
            # look at the last message and the one before that
            # if the role of both of them is the same
            # this is not valid
            # so we need to remove the last message
            last_role = messages[-1]['role']
            second_last_role = messages[-2]['role']
            if last_role == second_last_role:
                messages = messages[:-1]

            messages = [concat_text_lists(m) for m in messages]

            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            )

            print("Opened stream with model:", model_name)

            async def content_stream(original_stream):
                done_reasoning = False
                if reasoning:
                    yield '[{"reasoning": "'
                async for chunk in original_stream:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content'):
                        json_str = json.dumps(delta.reasoning_content)
                        without_quotes = json_str[1:-1]
                        yield without_quotes
                        print('\033[92m' + str(delta.reasoning_content) + '\033[0m', end='')
                    elif hasattr(delta, 'content'):
                        if reasoning and not done_reasoning:
                            yield '"}] <<CUT_HERE>>'
                            done_reasoning = True
                        yield delta.content or ""

            return content_stream(stream)

        except Exception as e:
            print('DeepSeek (OpenAI mode) error:', e)

    async def format_image_message(self, pil_image: Any, context: Any = None) -> dict:
        """Format image for DeepSeek using OpenAI's image format"""
        buffer = BytesIO()
        print('converting to base64')
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print('done')
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }

    async def chat(
        self,
        model: str,
        messages: list,
        context: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 5000
    ) -> str:
        """Non-streaming chat completion."""
        model_name = os.environ.get("AH_OVERRIDE_LLM_MODEL", "deepseek-chat")
        messages = [concat_text_lists(m) for m in messages]
        
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content

    async def get_service_models(self, context: Any = None) -> dict:
        """Get available models for DeepSeek."""
        return {
            "stream_chat": ["deepseek-chat", "deepseek-reasoner"],
            "chat": ["deepseek-chat", "deepseek-reasoner"]
        }
