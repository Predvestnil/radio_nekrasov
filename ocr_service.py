# ocr_service.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import asyncio
import time # For placeholder
import os # Added for file operations
import shutil # Added for directory cleanup
from PIL import Image # Added for image handling
from io import BytesIO # Added for image handling


app = FastAPI()


def timer(func):
    from functools import wraps
    from time import time
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"[{func.__name__!r}]: executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


MAX_SIZE_MB = 9.9
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024
RESIZE_STEP = 0.9

def _resize(filepath):
    original_size = os.path.getsize(filepath)
    if original_size <= MAX_SIZE_BYTES:
        return filepath

    print(f"original size: {original_size / (1024 * 1024):.2f} MB")

    img = Image.open(filepath)
    format = img.format

    quality = 95
    width, height = img.size
    while True:
        width = int(width * RESIZE_STEP)
        height = int(height * RESIZE_STEP)
        resized_img = img.resize((width, height))

        buffer = BytesIO()
        resized_img.save(buffer, format=format, quality=quality, optimize=True)
        size = buffer.tell()

        print(f"    trying size {width}x{height} — {size / (1024 * 1024):.2f} MB")

        if size <= MAX_SIZE_BYTES:
            output_path = f"{os.path.splitext(filepath)[0]}_resized.{format.lower()}" # Use original format
            with open(output_path, "wb") as f:
                f.write(buffer.getvalue())
            print(f"    saved as {output_path}")
            return output_path

# Вспомогательная функция для получения ответа от Yandex-OCR (взята из ocr.py)
def _get_response(file_path):
    import requests
    import json
    import base64

    def encode_file(file_path):
        with open(file_path, "rb") as fid:
            file_content = fid.read()
        return base64.b64encode(file_content).decode("utf-8")

    content = encode_file(file_path)

    data = {"mimeType": "JPG", # Assuming JPG, adjust if needed
            "languageCodes": ["ru"],
            "content": content}


    token = os.getenv("YANDEX_IAM_TOKEN", "YOUR_YANDEX_IAM_TOKEN")
    url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    headers= {"Content-Type": "application/json",
            "Authorization": "Bearer {:s}".format(token),
            "x-folder-id": os.getenv("YANDEX_FOLDER_ID", "YOUR_YANDEX_FOLDER_ID"),
            "model": "table",
            "x-data-logging-enabled": "true"}

    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    return response


def _process_response(response):
    import json
    try:
        data = json.loads(response.content.decode('utf-8'))
        results = []

        for block in data.get('result', {}).get('textAnnotation', {}).get('blocks', []):
            for line in block.get('lines', []):
                try:
                    vertices = line['boundingBox']['vertices']
                    bbox = {
                        'x1': int(vertices[0].get('x', 0)),
                        'y1': int(vertices[0].get('y', 0)),
                        'x2': int(vertices[2].get('x', 0)),
                        'y2': int(vertices[2].get('y', 0))
                    }
                    results.append({
                        'text': line.get('text', ''),
                        'bbox': bbox
                    })
                except (KeyError, IndexError) as e:
                    print(f"error while processing block: {e}")
                    continue

        return [result['text'] for result in results]

    except json.JSONDecodeError:
        print("error: response is not JSON")
        return []


def _clean_texts(texts):
    import re
    merged_text = ""

    for i, line in enumerate(texts):
        line = line.strip()

        if merged_text.endswith('-'):
            merged_text = merged_text[:-1] + line
        elif i > 0 and line and line[0].islower():
            merged_text += ' ' + line
        else:
            merged_text += '\n' + line

    merged_text = re.sub(r'\n+', '\n', merged_text)

    return merged_text.strip()


@timer
def ocr_fast(image_name):
    output_name = _resize(image_name)

    response = _get_response(output_name)
    texts_raw = _process_response(response)
    texts_cleaned = _clean_texts(texts_raw)

    # Вместо записи в файл, возвращаем текст
    f = open("texts_cleaned_fast.txt", "w")
    f.write(f'{response}\n{texts_raw}\n{texts_cleaned}')
    f.close()


    return texts_cleaned # Return the text


import numpy as np # Added for numpy usage
import cv2 # Added for cv2 usage
import torch # Added for torch usage
from doclayout_yolo import YOLOv10 # Added for YOLOv10

# Загружаем модель YOLOv10 один раз
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")

@timer
def _to_columns(points, x_eps = 10, y_eps = 20):
    i = 0
    tmp = list()
    while i < len(points) - 1:
        tmp.append([points[i]])
        for j in range(i+1, len(points)):
            if abs(points[i][0] - points[j][0]) < x_eps:
                tmp[-1].append(points[j])
            else:
                i = j
                break
        else:
            break

    tmp2 = list()
    for i in tmp:
        i = np.array(i)
        i = i[i[:, 1].argsort()]
        tmp2.append(i[0])
        for j in range(1, len(i)):
            if i[j][1] - tmp2[-1][3] < y_eps:
                tmp2[-1][0] = min([tmp2[-1][0], i[j][0]])
                tmp2[-1][1] = min([tmp2[-1][1], i[j][1]])
                tmp2[-1][2] = max([tmp2[-1][2], i[j][2]])
                tmp2[-1][3] = max([tmp2[-1][3], i[j][3]])
            else:
                tmp2.append(i[j])

    return np.array(tmp2)

@timer
def _page_layout(image_name):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'[INFO]: device = {device}')

    det_res = model.predict(
        image_name,
        imgsz=2048,
        conf=0.2,
        device=device
    )

    det_res = [el.cpu() for el in det_res]
    points = np.array(det_res[0].boxes.data)[det_res[0].boxes.cls == 1][:, :4]

    points = points[points[:, 1].argsort()]
    points = points[points[:, 0].argsort()]

    print(f'len of points = {len(points)}')

    columns = _to_columns(points)

    print(f'[INFO]: num of columns = {len(columns)}')
    return columns

@timer
def _save_bbox_crops(image_name, bboxes):
    print(f'[INFO] current image = {image_name}')

    output_dir = f'crops___{image_name}'
    img = cv2.imread(image_name)

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(crop_path, crop)

@timer
def _text_in_crops(image_name):
    crops = os.listdir('crops___' + image_name)
    page_crop_texts = []

    for crop in crops:
        response = _get_response(f'crops___{image_name}/{crop}')
        text_data = _process_response(response)
        if text_data: 
            merged_text = _clean_texts(text_data)
            if merged_text:
                page_crop_texts.append(merged_text)

    return page_crop_texts


@timer
def ocr_accurate(image_name):
    # Удаляем существующие временные директории перед созданием новых
    if os.path.exists('crops___' + image_name):
        shutil.rmtree('crops___' + image_name)
    os.mkdir('crops___' + image_name)

    page_layout = _page_layout(image_name)
    _save_bbox_crops(image_name, page_layout)
    texts_cleaned = _text_in_crops(image_name)

    # Сохраняем тексты в отдельные файлы в новой директории
    output_dir = f'accurate_texts___{image_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    output_files = []
    for i in range(len(texts_cleaned)):
        text = texts_cleaned[i]
        file_path = os.path.join(output_dir, f'{i}.txt')
        f = open(file_path, 'w')
        f.write(texts_cleaned[i])
        f.close()
        output_files.append(file_path)


    # Удаляем ненужную временную директорию
    shutil.rmtree('crops___' + image_name)

    return output_files # Return the list of output file paths

def image_from_bytes(uploaded_file: UploadFile):
    """
    Saves uploaded file bytes to a temporary file and returns the filename.
    """
    try:
        # Ensure the temp directory exists
        temp_dir = "/content/temp_uploaded_images"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())
        return file_path
    except Exception as e:
        print(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Error saving uploaded file")



@app.post("/process")
async def process_image(image: UploadFile = File(...), mode: str = "fast"):
    """
    Accepts an image and returns extracted text based on the specified mode ('fast' or 'accurate').
    """
    try:
        if not image.filename:
             raise HTTPException(status_code=400, detail="No file uploaded")
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
             raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, and JPEG are supported.")


        print(f"OCR Service: Received image {image.filename}, size {image.size}, mode: {mode}")

        # Save the uploaded file temporarily
        image_path = image_from_bytes(image)

        if mode == "fast":
            extracted_text = ocr_fast(image_path)
            return {"mode": "fast", "text": extracted_text}
        elif mode == "accurate":
            output_files = ocr_accurate(image_path)
             # Clean up the temporary image file
            os.remove(image_path)
            # In accurate mode, text is saved to files. We return the list of file paths.
            return {"mode": "accurate", "output_files": output_files}
        else:
            # Clean up the temporary image file if mode is invalid
            os.remove(image_path)
            raise HTTPException(status_code=400, detail="Invalid mode. Choose 'fast' or 'accurate'.")

    except HTTPException as he:
        # Clean up the temporary image file if an HTTPException occurs
        if 'image_path' in locals() and os.path.exists(image_path):
             os.remove(image_path)
        raise he # Re-raise client/server errors we explicitly created
    except Exception as e:
        # Clean up the temporary image file if any other exception occurs
        if 'image_path' in locals() and os.path.exists(image_path):
             os.remove(image_path)
        print(f"OCR Service Error processing {image.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during OCR: {e}")

if __name__ == "__main__":
    # Ensure you run this on a different port, e.g., 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)