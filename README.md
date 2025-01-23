# MindRoot Gemini Plugin

A plugin for integrating Google's Gemini LLM models into MindRoot, with support for streaming, multimodal inputs, and caching.

## Features

- Streaming text generation
- Multimodal support (text + images)
- Message caching system
- Configurable model selection
- Debug output support

## Installation

```bash
pip install -e .
```

## Configuration

The plugin requires the following environment variables:

- `GOOGLE_API_KEY`: Your Google API key for accessing Gemini models
- `DEFAULT_LLM_MODEL` (optional): Override the default model (defaults to "gemini-exp-1206")
- `MR_DEBUG` (optional): Enable debug output when set to "True"

## Available Models

- `gemini-exp-1206` (default) - Base experimental model
- `gemini-2.0-flash-exp` - Flash experimental model
- `gemini-2.0-flash-thinking-exp-1219` - Flash model with thinking mode

## Services

### stream_chat
Streams text completions from Gemini models with support for:
- Temperature control
- Max token limits
- Message caching
- Multimodal inputs (text + images)
- Debug output

### format_image_message
Formats images for Gemini's multimodal input:
- Converts to base64
- Proper MIME type handling (image/png)
- Optimized for Gemini's content format

### get_image_dimensions
Returns supported image dimensions for Gemini models:
- Max width: 4096px
- Max height: 4096px
- Max pixels: 16,777,216

## Usage Example

```python
# Text only
response = await stream_chat(model="gemini-exp-1206", messages=[
    {"role": "user", "content": "Hello, how are you?"}
])

# With image
from PIL import Image
image = Image.open("example.png")
image_content = await format_image_message(image)

response = await stream_chat(model="gemini-exp-1206", messages=[
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        image_content
    ]}
])
```

## Development

The plugin follows the MindRoot plugin architecture and includes:
- Message caching system
- Debug output capabilities
- Error handling
- Environment variable configuration

## License

MIT License
