import json
import base64
from pathlib import Path
from aime_api_client_interface import do_api_request

def generate_image():
    # Define the image generation parameters
    params = {
        'prompt': 'Astronaut on Mars holding a banner which states "AIME is happy to serve your model" during sunset sitting on a giant yellow rubber duck',
        'seed': -1,
        'height': 1024,
        'width': 1024,
        'steps': 50,
        'guidance': 3.5,
        'image2image_strength': 0.8,
        'provide_progress_images': 'none',
        'wait_for_result': True
    }

    # Call the AIME API
    final = do_api_request(
        'https://api.aime.info',
        'flux-dev',
        params,
        user='apiexample@aime.info',
        key='181e35ac-7b7d-4bfe-9f12-153757ec3952'
    )

    # Save the images
    images = final.get('images') or final.get('job_result', {}).get('images', [])
    if not images:
        print("No images returned by the API.")
        return final
    for i, img_b64 in enumerate(images):
        header, img_data = img_b64.split(',', 1) if ',' in img_b64 else (None, img_b64)
        img_bytes = base64.b64decode(img_data)
        filename = Path(__file__).parent / f'image_{i}.png'
        filename.write_bytes(img_bytes)
        print(f"Saved image to: {filename}")
    print(f"\nImage generation complete. {len(images)} image(s) saved.")
    return final

if __name__ == "__main__":
    generate_image()