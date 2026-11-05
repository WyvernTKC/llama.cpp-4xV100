/**
 * Creates a base64 data URL from MIME type and base64-encoded data.
 *
 * @param mimeType - The MIME type (e.g., 'image/png', 'audio/mp3')
 * @param base64Data - The base64-encoded data
 * @returns A data URL string in format 'data:{mimeType};base64,{data}'
 */
export function createBase64DataUrl(mimeType: string, base64Data: string): string {
	return `data:${mimeType};base64,${base64Data}`;
}

/**
 * Saves a data URL as a file. The bytes go through a blob URL, which every
 * browser downloads regardless of size.
 *
 * @param dataUrl - The data URL to save
 * @param filename - Suggested filename
 */
export async function downloadDataUrl(dataUrl: string, filename: string): Promise<void> {
	const blob = await (await fetch(dataUrl)).blob();
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');

	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}
