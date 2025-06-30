export async function processImageFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function processPDFFile(file: File): Promise<string> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    
    // Simple PDF text extraction using basic parsing
    const decoder = new TextDecoder('utf-8');
    let text = decoder.decode(uint8Array);
    
    // Extract text between stream objects (basic PDF parsing)
    const textMatches = text.match(/stream\s*(.*?)\s*endstream/gs);
    let extractedText = '';
    
    if (textMatches) {
      textMatches.forEach(match => {
        const content = match.replace(/stream\s*|\s*endstream/g, '');
        // Try to extract readable text
        const readableText = content.replace(/[^\x20-\x7E\n\r\t]/g, ' ').trim();
        if (readableText.length > 10) {
          extractedText += readableText + '\n';
        }
      });
    }
    
    // Fallback: try to extract any readable text from the entire PDF
    if (!extractedText.trim()) {
      const allText = text.replace(/[^\x20-\x7E\n\r\t]/g, ' ');
      const words = allText.split(/\s+/).filter(word => 
        word.length > 2 && /^[a-zA-Z0-9]/.test(word)
      );
      extractedText = words.slice(0, 1000).join(' '); // Limit to first 1000 words
    }
    
    return extractedText.trim() || 'Unable to extract text from this PDF. The document may be image-based or encrypted.';
  } catch (error) {
    console.error('Error processing PDF:', error);
    return 'Error processing PDF file. Please try a different document or ensure the PDF is not corrupted.';
  }
}

export async function processTextFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/');
}

export function isPDFFile(file: File): boolean {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

export function isTextFile(file: File): boolean {
  return file.type.startsWith('text/') || 
         file.name.endsWith('.md') || 
         file.name.endsWith('.txt') ||
         file.name.endsWith('.doc') ||
         file.name.endsWith('.docx') ||
         file.type === 'application/msword' ||
         file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}