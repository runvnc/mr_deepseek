from lib.providers.services import service
import google.generativeai as genai
import os
import base64
from io import BytesIO
import json
from .message_utils import compare_messages

# Initialize Gemini client
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Default to experimental model or env override
DEFAULT_MODEL = "gemini-exp-1206"

# Store last messages for caching
_last_messages = []

def prepare_message_content(message):
    """Convert message content to Gemini format"""
    msg_copy = dict(message)
    if isinstance(msg_copy.get('content'), str):
        msg_copy['content'] = [{
            "type": "text",
            "text": msg_copy['content']
        }]
    return msg_copy

@service()
async def stream_chat(model, messages=[], context=None, num_ctx=200000, 
                     temperature=0.0, max_tokens=2500, num_gpu_layers=0):
    try:
        global _last_messages
        print("gemini stream_chat")
        messages = [dict(message) for message in messages]
        print('\033[93m' + '-'*80 + '\033[0m')
        
        # Use env model or default
        model_name = os.environ.get("DEFAULT_LLM_MODEL", DEFAULT_MODEL)
        model = genai.GenerativeModel(model_name)
        
        # Format messages
        formatted_messages = [prepare_message_content(msg) for msg in messages[1:]]
        
        # Compare with last messages for caching
        changed_indices = compare_messages(_last_messages, formatted_messages)
        _last_messages = formatted_messages.copy()

        # Debug output
        if os.environ.get('MR_DEBUG') == 'True':
            print('\033[93m' + 'formatted_messages' + '\033[0m')
            print(json.dumps(formatted_messages, indent=4))

        # Prepare content parts list for Gemini
        content_parts = []
        for msg in formatted_messages:
            for content in msg['content']:
                if content['type'] == 'text':
                    content_parts.append(content['text'])
                elif content['type'] == 'image':
                    # For image content, we need to pass the base64 data directly
                    content_parts.append({
                        "mime_type": content['source']['media_type'],
                        "data": content['source']['data']
                    })

        # Create streaming response
        response = model.generate_content(
            content_parts,
            stream=True,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )

        async def content_stream():
            async for chunk in response:
                if chunk.text:
                    if os.environ.get('MR_DEBUG') == 'True':
                        print('\033[92m' + chunk.text + '\033[0m', end='')
                    yield chunk.text
                else:
                    if os.environ.get('MR_DEBUG') == 'True':
                        print('\033[93m' + '-'*80 + '\033[0m')
                        print('\033[93m' + str(chunk) + '\033[0m')
                    yield ''

        return content_stream()

    except Exception as e:
        print('Gemini error:', e)
        raise

@service()
async def format_image_message(pil_image, context=None):
    """Format image for Gemini multimodal input"""
    buffer = BytesIO()
    print('converting to base64')
    pil_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    print('done')
    
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_base64
        }
    }

@service()
async def get_image_dimensions(context=None):
    """Return max supported image dimensions for Gemini"""
    return 4096, 4096, 16777216  # Max width, height, pixels
