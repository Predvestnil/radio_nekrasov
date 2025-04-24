# Некрасовка Daily

## Установка (Linux):

*Требуется выполнить следующие команды в командой строке:*
- `git clone https://github.com/Predvestnil/radio_nekrasov.git`
- `cd radio_nekrasov`
- `pip install -r requirements.txt`
- `wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip`
- `unzip -o ngrok-stable-linux-amd64.zip`
- `wget https://huggingface.co/breezedeus/pix2text-layout-docyolo/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt`
- `nohup python /content/server.py & python /content/llm_service.py & python /content/ocr_service.py & python /content/tts_service.py`

---

## Тестирование

- После выполнения всех команд перейдите по ссылке `localhost:3000`
- Для доступа из интернета можно: 
- скачать [ngrok](https://www.ngrok.com/), выполнить команду `ngrok http 3000`, скопировать полученный ip адрес(_не работает с ru адресов_); 
- приобрести статический ip адрес и изменить в файле `server.py` строку `uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=True)` на строку `uvicorn.run("server:app", host="{ваш айпи}", port=3000, reload=True)`

---

## Пример работы сервиса

- Посмотреть пример работы сервиса можно в [корневой папке репозитория]()

---

## Минимальные требования к оборудованию
- CPU 4 ядра
- GPU 16Gb VRAM
- 16 GB RAM
