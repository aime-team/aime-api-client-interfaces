const { doAPIRequest } = require('../model_api');
const fs = require('fs');

// Configuration
const CONFIG = {
    BASE_URL: 'https://api.aime.info',
    ENDPOINT: 'flux-dev',
    API_EMAIL: 'apiexample@aime.info',
    API_KEY: '181e35ac-7b7d-4bfe-9f12-153757ec3952'
};

// Set up global fetch to handle base URL
const { fetch: originalFetch } = global;
global.fetch = (input, init = {}) => {
    const url = typeof input === 'string' 
        ? input.startsWith('http') ? input : `${CONFIG.BASE_URL}${input.startsWith('/') ? '' : '/'}${input}`
        : input;
    return originalFetch(url, init);
};

// Image generation parameters
const params = {
    prompt: 'A beautiful sunset over mountains',
    height: 1024,
    width: 1024,
    steps: 30,
    guidance: 7.5,
    wait_for_result: true
};

console.log('🚀 Starting image generation...');

// Make the API request
doAPIRequest(
    CONFIG.ENDPOINT,
    params,
    (result, error) => {
        if (error) {
            console.error('❌ Error:', error.message || 'Unknown error');
            return;
        }

        const images = result.images || [];
        if (images.length === 0) {
            console.log('⚠️  No images were generated');
            return;
        }

        // Save the first image
        const imageData = images[0].includes(',') ? images[0].split(',')[1] : images[0];
        const outputPath = 'generated_image.png';
        fs.writeFileSync(outputPath, Buffer.from(imageData, 'base64'));
        
        console.log(`✅ Image saved as ${outputPath}`);
        console.log('✨ All done!');
    },
    CONFIG.API_EMAIL,
    CONFIG.API_KEY,
    (progress) => {
        const msg = progress.progress >= 0 
            ? `Progress: ${progress.progress}%` 
            : 'Starting...';
        process.stdout.write(`\r${msg}${' '.repeat(20)}`);
    }
);
