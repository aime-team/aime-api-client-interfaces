const { doAPIRequest } = require('../model_api');
const fs = require('fs');

// Configuration
const CONFIG = {
    BASE_URL: 'https://api.aime.info',
    ENDPOINT: 'tts_tortoise',
    API_EMAIL: 'apiexample@aime.info',
    API_KEY: '181e35ac-7b7d-4bfe-9f12-153757ec3952',
    OUTPUT_FILE: 'generated_speech.mp3'
};

// Set up global fetch to handle base URL
const { fetch: originalFetch } = global;
global.fetch = (input, init = {}) => {
    const url = typeof input === 'string' 
        ? input.startsWith('http') ? input : `${CONFIG.BASE_URL}${input.startsWith('/') ? '' : '/'}${input}`
        : input;
    return originalFetch(url, init);
};

// TTS parameters
const params = {
    text: "Hello, this is a test of the text-to-speech system. How are you doing today?",
    voice: "emma",
    preset: "fast",
};

console.log('🔊 Starting text-to-speech generation...');

// Make the API request
doAPIRequest(
    CONFIG.ENDPOINT,
    params,
    (result, error) => {
        if (error) {
            console.error('❌ Error:', error.message || 'Unknown error');
            return;
        }

        // Check for audio data in both audio_output and audio_data fields
        const audioData = result.audio_output || result.audio_data;
        
        if (audioData) {
            // Extract base64 data if it's a data URL
            const base64Data = audioData.includes(',') 
                ? audioData.split(',')[1] 
                : audioData;
            
            try {
                fs.writeFileSync(CONFIG.OUTPUT_FILE, Buffer.from(base64Data, 'base64'));
                console.log(`✅ Audio saved as ${CONFIG.OUTPUT_FILE}`);
                return;
            } catch (writeError) {
                console.error('❌ Error saving audio file:', writeError.message);
            }
        }
        
        console.log('⚠️  No audio data received in the response');
        if (result) {
            console.log('API response:', JSON.stringify(result, null, 2));
        }
        
        console.log('✨ All done!');
    },
    CONFIG.API_EMAIL,
    CONFIG.API_KEY,
    (progress) => {
        const msg = progress.progress >= 0 
            ? `Progress: ${progress.progress}%` 
            : 'Processing...';
        process.stdout.write(`\r${msg}${' '.repeat(20)}`);
    }
);
