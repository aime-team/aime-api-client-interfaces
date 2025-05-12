import asyncio
import base64
from aime_api_client_interface import ModelAPI

def save_audio(audio_base64: str, output_filename: str = "output.wav"):
    audio_data = base64.b64decode(audio_base64)
    with open(output_filename, "wb") as f:
        f.write(audio_data)
    print(f"Saved audio to: {output_filename}")

def progress_callback(progress_info, progress_data):
    if progress_info:
        print(f"Progress: {progress_info}")
    if progress_data:
        print(f"Progress data: {progress_data}")

async def main():
    model_api = ModelAPI('https://api.aime.info', 'tts_tortoise', 'apiexample@aime.info', '181e35ac-7b7d-4bfe-9f12-153757ec3952')

    await model_api.do_api_login_async()

    params = {
        "text": "This is an asynchronous text to speech example.",
        "language": "eng",
        "voice": "emma"
    }
    async def result_callback(result):
        if result and 'audio' in result:
            save_audio(result['audio'], "output_async.wav")
    
    await model_api.do_api_request_async(
        params,
        result_callback=result_callback,
        progress_callback=progress_callback
    )

    await model_api.close_session()

if __name__ == "__main__":
    asyncio.run(main())