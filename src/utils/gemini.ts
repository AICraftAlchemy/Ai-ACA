import { GoogleGenerativeAI } from '@google/generative-ai';

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;

if (!API_KEY) {
  throw new Error('VITE_GEMINI_API_KEY is not set in environment variables');
}

const genAI = new GoogleGenerativeAI(API_KEY);

export async function generateResponse(
  prompt: string,
  fileData?: { mimeType: string; data: string },
  conversationHistory: Array<{ role: string; parts: Array<{ text: string }> }> = []
): Promise<string> {
  try {
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-2.0-flash-exp',
      generationConfig: {
        maxOutputTokens: 32768, // Increased token limit for longer responses
        temperature: 0.7,
        topP: 0.8,
        topK: 40,
      },
    });

    const contents = [
      ...conversationHistory,
      {
        role: 'user',
        parts: fileData 
          ? [
              { text: prompt },
              {
                inlineData: {
                  mimeType: fileData.mimeType,
                  data: fileData.data
                }
              }
            ]
          : [{ text: prompt }]
      }
    ];

    const result = await model.generateContentStream({
      contents,
    });

    let response = '';
    for await (const chunk of result.stream) {
      const chunkText = chunk.text();
      response += chunkText;
    }

    return response.trim();
  } catch (error) {
    console.error('Error generating response:', error);
    if (error instanceof Error) {
      if (error.message.includes('API key')) {
        throw new Error('Invalid API key. Please check your configuration.');
      } else if (error.message.includes('quota')) {
        throw new Error('API quota exceeded. Please try again later.');
      } else if (error.message.includes('safety')) {
        throw new Error('Content was blocked by safety filters. Please rephrase your request.');
      }
    }
    throw new Error('Failed to generate response. Please try again.');
  }
}

export async function fileToGenerativePart(file: File): Promise<{ mimeType: string; data: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64Data = (reader.result as string).split(',')[1];
      resolve({
        mimeType: file.type,
        data: base64Data
      });
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}