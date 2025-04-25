# server.py
import os
import json
from fastapi import FastAPI, Request, HTTPException, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import httpx
from config import OCR_SERVICE_URL, LLM_SERVICE_URL, TTS_SERVICE_URL, GENERATED_AUDIO_DIR, GENERATED_AUDIO_STATIC_PATH
import time
import uuid

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount generated audio directory
os.makedirs(os.path.join("static", GENERATED_AUDIO_DIR), exist_ok=True)
app.mount(GENERATED_AUDIO_STATIC_PATH, StaticFiles(directory=os.path.join("static", GENERATED_AUDIO_DIR)), name="generated_audio")

templates = Jinja2Templates(directory="templates")

DATA_FILE = "audio_data.json"
SAMPLES_DIR = "static/samples"
UPLOAD_DIR = "static/uploads"

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def load_audio_data():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: {DATA_FILE} not found or invalid. Starting with empty data.")
        return []

def save_audio_data(data):
    """Saves the audio data structure back to the JSON file."""
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving audio data to {DATA_FILE}: {e}")



async def extract_text_from_single_file(file: UploadFile, http_client: httpx.AsyncClient) -> dict:
    """Extract text from a single image file using OCR service."""
    filename = file.filename
    print(f"WebApp: Extracting text from file: {filename}")
    content = await file.read()
    
    if not content:
        return {"success": False, "message": f"Skipped {filename}: Empty file.", "text": ""}
    
    try:
        # Call OCR Service
        files = {'image': (filename, content, file.content_type)}
        ocr_response = await http_client.post(f"{OCR_SERVICE_URL}/process", files=files, timeout=30.0)
        ocr_response.raise_for_status()
        ocr_result = ocr_response.json()
        extracted_text = ocr_result.get("text", "")
        
        if not extracted_text:
            print(f"WebApp: OCR returned no text for {filename}")
            return {"success": False, "message": f"Skipped {filename}: OCR failed or returned no text.", "text": ""}
        
        print(f"WebApp: OCR successful for {filename}. Text length: {len(extracted_text)}")
        return {"success": True, "message": f"Extracted text from {filename}", "text": extracted_text}
    
    except Exception as e:
        print(f"WebApp: Error extracting text from {filename}: {e}")
        return {"success": False, "message": f"Failed to extract text from {filename}: {str(e)}", "text": ""}


@app.post("/upload")
async def upload_article(files: Optional[List[UploadFile]] = File(None),
                         samples: Optional[List[str]] = Form(None),
                         model: str = Form(...)):
    all_extracted_texts = []
    processing_results = []
    file_count = 0
    
    async with httpx.AsyncClient() as client:
        

        if files:
            for file in files:
                allowed_extensions = ('.png', '.jpg', '.jpeg', '.pdf')
                if file.filename and file.filename.lower().endswith(allowed_extensions):
                    result = await extract_text_from_single_file(file, client)
                    processing_results.append(result["message"])
                    if result["success"]:
                        all_extracted_texts.append(result["text"])
                        file_count += 1
                else:
                    processing_results.append(f"Skipped {file.filename or 'unnamed file'}: Not a supported image type.")


        if samples:
            for sample_name in samples:
                sample_path = f"{SAMPLES_DIR}/{sample_name}"
                if os.path.exists(sample_path):
                    try:
                        with open(sample_path, "rb") as f:
                            content_type = "image/jpeg"
                            if sample_name.lower().endswith(".png"):
                                content_type = "image/png"
                            
                            from io import BytesIO
                            file_content = BytesIO(f.read())
                            mock_file = UploadFile(filename=sample_name, file=file_content)
                            print(f"WebApp: Processing sample: {sample_name}")
                            
                            result = await extract_text_from_single_file(mock_file, client)
                            processing_results.append(result["message"])
                            if result["success"]:
                                all_extracted_texts.append(result["text"])
                                file_count += 1
                    except Exception as e:
                        print(f"WebApp: Error reading or processing sample {sample_name}: {e}")
                        processing_results.append(f"Failed processing sample {sample_name}: Error reading file.")
                else:
                    processing_results.append(f"Skipped sample {sample_name}: File not found.")
        

        if not all_extracted_texts:
            return JSONResponse(content={"result": "No text could be extracted from the provided files."}, status_code=400)
        
        # Second step: Concatenate all texts and process through LLM and TTS
        concatenated_text = "\n\n".join(all_extracted_texts)
        print(f"WebApp: Concatenated text from {file_count} files. Total length: {len(concatenated_text)}")
        
        try:
            # Process through LLM
            print(f"WebApp: Sending concatenated text to LLM (model: {model})...")
            llm_payload = {"text": concatenated_text, "model": model}
            llm_response = await client.post(f"{LLM_SERVICE_URL}/format", json=llm_payload, timeout=120.0)
            llm_response.raise_for_status()
            llm_result = llm_response.json()
            formatted_text = llm_result.get("formatted_text")
            
            if not formatted_text:
                print("WebApp: LLM returned no formatted text")
                return JSONResponse(content={"result": "LLM formatting failed or returned no text."}, status_code=500)
            
            print(f"WebApp: LLM formatting successful. Text length: {len(formatted_text)}")
            
            # Process through TTS
            print("WebApp: Sending formatted text to TTS service...")
            tts_payload = {"text": formatted_text}
            print(tts_payload)
            tts_response = await client.post(f"{TTS_SERVICE_URL}/synthesize", json=tts_payload, timeout=60.0)
            tts_response.raise_for_status()
            tts_result = tts_response.json()
            audio_path = tts_result.get("audio_path")
            
            if not audio_path:
                print("WebApp: TTS returned no audio path")
                return JSONResponse(content={"result": "TTS synthesis failed."}, status_code=500)
            
            print(f"WebApp: TTS successful. Audio path: {audio_path}")
            
            # Add final success message
            processing_results.append(f"Successfully processed {file_count} files into a single audio file.")
            
            return JSONResponse(content={
                "result": "; ".join(processing_results),
                "audio_path": audio_path,
                "files_processed": file_count
            })
            
        except httpx.HTTPStatusError as e:
            err_detail = "Unknown error"
            try:
                err_detail = e.response.json().get("detail", e.response.text)
            except:
                err_detail = e.response.text
            print(f"WebApp: HTTP Error in batch processing: Status {e.response.status_code} from {e.request.url} - {err_detail}")
            return JSONResponse(content={"result": f"Error from {e.request.url.path} service - {e.response.status_code}"}, status_code=500)
        except Exception as e:
            print(f"WebApp: Unexpected Error in batch processing: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(content={"result": f"Unexpected internal error: {str(e)}"}, status_code=500)



@app.get("/api/years", response_class=JSONResponse)
async def get_years():
    data = load_audio_data()
    # Handle potential empty data or incorrect structure gracefully
    years = []
    if isinstance(data, list):
        for item in data:
             if isinstance(item, dict) and item:
                 years.append(list(item.keys())[0])
    return {"years": sorted(list(set(years)))} # Sort and ensure unique

@app.get("/api/months/{year}", response_class=JSONResponse)
async def get_months(year: str):
    data = load_audio_data()
    months = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and year in item:
                year_data = item[year]
                if isinstance(year_data, list):
                    for month_item in year_data:
                        if isinstance(month_item, dict) and month_item:
                             months.append(list(month_item.keys())[0])
                    # Found the year, no need to continue looping through top-level list
                    return {"months": sorted(list(set(months)))} # Sort and unique
    # If loop finishes without finding the year
    raise HTTPException(status_code=404, detail=f"Year {year} not found or has no month data")


@app.get("/api/audio/{year}/{month}", response_class=JSONResponse)
async def get_audio_path(year: str, month: str):
    data = load_audio_data()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and year in item:
                year_data = item[year]
                if isinstance(year_data, list):
                    for month_item in year_data:
                        if isinstance(month_item, dict) and month in month_item:
                            path = month_item[month]
                            # Basic validation: check if path starts reasonably
                            if isinstance(path, str) and path.startswith("/static/"):
                                return {"path": path}
                            else:
                                print(f"Warning: Invalid path found for {month} {year}: {path}")
                                # Fall through to raise 404 if no valid path found
    # If loops finish without finding a valid path
    raise HTTPException(status_code=404, detail=f"Audio for {month} {year} not found or path invalid")


# --- Keep HTML serving and Sample Gallery endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/samples", response_class=JSONResponse)
async def get_samples():
    samples = []
    valid_image_extensions = ('.jpg', '.jpeg', '.png')
    try:
        if not os.path.isdir(SAMPLES_DIR):
             print(f"Warning: Samples directory '{SAMPLES_DIR}' not found.")
             return {"samples": []}

        for filename in os.listdir(SAMPLES_DIR):
            filepath = os.path.join(SAMPLES_DIR, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(valid_image_extensions):
                name_part, ext = os.path.splitext(filename)
                thumbnail_filename = f"{name_part}_thumb{ext}"
                thumbnail_path = f"/static/samples/{thumbnail_filename}"

                samples.append({
                    "name": filename,
                    "path": f"/static/samples/{filename}",
                    "thumbnail": thumbnail_path,
                    "type": "image"
                })
    except Exception as e:
        print(f"Error listing samples in '{SAMPLES_DIR}': {e}")
        return {"samples": []}

    return {"samples": samples}


if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    from pyngrok import ngrok

    ngrok_tunnel = ngrok.connect(3000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=True)