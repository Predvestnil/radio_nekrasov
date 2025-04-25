# llm_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from abc import ABC, abstractmethod
import os
import asyncio # For placeholder simulation

# --- LLM Configuration 
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "YOUR_YANDEX_API_KEY")
YANDEX_SERVICE_ID = os.getenv("YANDEX_SERVICE_ID", "YOUR_YANDEX_SERVICE_ID")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "YOUR_YANDEX_FOLDER_ID")
YANDEX_PRIVATE_KEY = os.getenv("YANDEX_PRIVATE_KEY", "YOUR_YANDEX_PRIVATE_KEY")
YANDEX_API_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion" # Example


LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "llama3") # Example for local Ollama
# --- End Configuration ---

app = FastAPI()

# --- LLM Formatter Interface and Implementations ---
class LLMFormatter(ABC):
    @abstractmethod
    async def format(self, text: str) -> str:
        pass

class YandexGPTFormatter(LLMFormatter):
    async def format(self, text: str) -> str:
        """Placeholder: Implement actual YandexGPT API call."""
        import requests
        import json
        import time
        import jwt
        import os
        service_account_id = YANDEX_SERVICE_ID
        key_id = YANDEX_FOLDER_ID
        now = int(time.time())
        payload = {
                'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
                'iss': service_account_id,
                'iat': now,
                'exp': now + 360}
        sa_key = YANDEX_PRIVATE_KEY
        # Формирование JWT
        encoded_token = jwt.encode(
            payload,
            sa_key,
            algorithm='PS256',
            headers={'kid': YANDEX_FOLDER_ID})

        url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
        x = requests.post(url,  headers={'Content-Type': 'application/json'}, json = {'jwt': encoded_token}).json()
        token = x['iamToken']
        import requests


        catalog_id = YANDEX_FOLDER_ID
        API_KEY = YANDEX_API_KEY
        magazine_text = text

        prompt_text = f'''
            Ниже — шаблон промпта для LLM  который на вход принимает OCR‑текст газет и дату, а на выходе выдаёт готовый к передаче в Bark TTS сценарий «Некрасовка daily». В нём оставлены маркеры для подстановки ваших данных:
        Учти, что тексты, полученные OCR с газеты среднего качества, поэтому постарайся восполнить смысловую часть текста из газеты.
        Веди повествование так  ,как будто эфир проводится в эпоху выхода газеты
        # Настройки модели
        Использовать tts Bark от suno‑ai.  
        Объявить спикеров и их голоса:
        speaker_lookup = {{
            "Samantha": "v2/ru_speaker_5",   FEMALE
            "John":     "v2/ru_speaker_3"     MALE
        }}

        # Входные данные 
        Дата: {{DATE}}            # формат: DD Month YYYY, например, 22 April 2025
        OCR‑текст газет: {{NEWSPAPER_TEXT}}

        # Задача
        Сгенерировать полный текст сценария радиопередачи «Некрасовка daily», который сразу можно передать в Bark TTS.  
        Структура должна быть строго выдержана:

        1. **Вводная секция**  
        - Уникальный приветственный текст (каждый раз — новый), озвучиваемый ведущими.  
        - Используйте теги для эффектов: `[clears throat]`, `[laughter]`, `—` (для пауз) и др.

        2. **«В этот день … назад»**  
        - Вырезки из архивов `{{NEWSPAPER_TEXT}}` за ту же дату 2, 5, 10… лет назад.  
        - Каждый факт — отдельная мини‑зарисовка, озвучиваемая одним из ведущих.

        3. **Тематические подборки**  
        - **Новости страны**  
        - **Достижения** (промышленность, сельское хозяйство, космос…)  
        - **События культуры** (книги, кино, музыка)  
        - **Новости спорта**  
        - **Разное** (юмор, объявления и т.п.)  

        Для каждого блока:  
        - Подберите и отредактируйте из `{{NEWSPAPER_TEXT}}` 2–3 цитаты/факта.  
        - Обязательно плавные переходы между блоками (написать короткий связующий текст).

        4. **Музыкальные вставки [опционально]**  
        - В нужных местах добавить тег `[music]…[/music]` или просто `[music]`.

        5. **Форматирование диалога**  
        Всю передачу оформить как череду реплик двух ведущих, пример оформления ниже:
        [WOMAN] Samantha: Приветствую вас в «Некрасовка daily»! [clears throat] [MAN] John: Добрый день, дорогие слушатели! Сегодня… [laughs]

        – переключайтесь между спикерами по необходимости.

        6. **Теги suno bark‑ai**  
        Обязательно использовать:
        - `[laughter]`, `[laughs]`, `[clears throat]`  
        - Дефис–тире (`—`) для естественных пауз  
        - `— or …` для заминок речи  
        - `♪ … ♪` для песенных фрагментов  
        - `CAPITALIZATION` для ударения слов
        d 
        7. Добавь 2-3 раза в некоторые моменты передачи тег [interrupt-podcast], на это место будет вставлена музыка-перебивка (как в радио эфирах)

        # Ожидаемый результат
        Готовый текстовый сценарий радиопередачи «Некрасовка daily» с чётким разделением на секции, спикерами и эффектами — полностью готовый к передаче на вход Bark TTS.

        {{DATE}}  = ищи дату в тексте
        {{NEWSPAPER_TEXT}} = {magazine_text}
        '''
        prompt = {
            "modelUri": f"gpt://{catalog_id}/yandexgpt/rc",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,
                "maxTokens": "100000"
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt_text
                }
            ]
        }


        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {API_KEY}"
        }
        # Адрес для обращения к модели 

        url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'

        data = {}

        # Указываем тип модели
        #'gpt://<идентификатор_каталога>/yandexgpt-lite'
        data['modelUri'] = f"gpt://b1g9016l80j9p7jb12rq/yandexgpt/rc"

        # Настраиваем дополнительные параметры модели
        data['completionOptions'] = {'stream': False,
                                    'temperature': 0.3,
                                    'maxTokens': 100000}

        # Указываем контекст для модели
        data['messages'] =  [
                {
                    "role": "user",
                    "text": prompt_text
                }
            ]

        # Получаем ответ модели
        response = requests.post(url, headers={'Authorization': 'Bearer ' + token}, json = data).json()
        return response['result']['alternatives'][0]['message']['text']

class LlamaFormatter(LLMFormatter):
    async def format(self, text: str) -> str:
        """Placeholder: Implement actual YandexGPT API call."""
        import requests
        import json
        import time
        import jwt
        import os
        service_account_id = YANDEX_SERVICE_ID
        key_id = YANDEX_FOLDER_ID
        now = int(time.time())
        payload = {
                'aud': 'https://iam.api.cloud.yandex.net/iam/v1/tokens',
                'iss': service_account_id,
                'iat': now,
                'exp': now + 360}
        sa_key = YANDEX_PRIVATE_KEY
        # Формирование JWT
        encoded_token = jwt.encode(
            payload,
            sa_key,
            algorithm='PS256',
            headers={'kid': key_id})

        url = 'https://iam.api.cloud.yandex.net/iam/v1/tokens'
        x = requests.post(url,  headers={'Content-Type': 'application/json'}, json = {'jwt': encoded_token}).json()
        token = x['iamToken']
        import requests


        catalog_id = key_id
        API_KEY = YANDEX_API_KEY
        magazine_text = text

        prompt_text = f'''
            Ниже — шаблон промпта для LLM  который на вход принимает OCR‑текст газет и дату, а на выходе выдаёт готовый к передаче в Bark TTS сценарий «Некрасовка daily». В нём оставлены маркеры для подстановки ваших данных:
        Учти, что тексты, полученные OCR с газеты среднего качества, поэтому постарайся восполнить смысловую часть текста из газеты.
        Веди повествование так  ,как будто эфир проводится в эпоху выхода газеты
        # Настройки модели
        Использовать tts Bark от suno‑ai.  
        Объявить спикеров и их голоса:
        speaker_lookup = {{
            "Samantha": "v2/ru_speaker_5",   FEMALE
            "John":     "v2/ru_speaker_3"     MALE
        }}

        # Входные данные 
        Дата: {{DATE}}            # формат: DD Month YYYY, например, 22 April 2025
        OCR‑текст газет: {{NEWSPAPER_TEXT}}

        # Задача
        Сгенерировать полный текст сценария радиопередачи «Некрасовка daily», который сразу можно передать в Bark TTS.  
        Структура должна быть строго выдержана:

        1. **Вводная секция**  
        - Уникальный приветственный текст (каждый раз — новый), озвучиваемый ведущими.  
        - Используйте теги для эффектов: `[clears throat]`, `[laughter]`, `—` (для пауз) и др.

        2. **«В этот день … назад»**  
        - Вырезки из архивов `{{NEWSPAPER_TEXT}}` за ту же дату 2, 5, 10… лет назад.  
        - Каждый факт — отдельная мини‑зарисовка, озвучиваемая одним из ведущих.

        3. **Тематические подборки**  
        - **Новости страны**  
        - **Достижения** (промышленность, сельское хозяйство, космос…)  
        - **События культуры** (книги, кино, музыка)  
        - **Новости спорта**  
        - **Разное** (юмор, объявления и т.п.)  

        Для каждого блока:  
        - Подберите и отредактируйте из `{{NEWSPAPER_TEXT}}` 2–3 цитаты/факта.  
        - Обязательно плавные переходы между блоками (написать короткий связующий текст).

        4. **Музыкальные вставки [опционально]**  
        - В нужных местах добавить тег `[music]…[/music]` или просто `[music]`.

        5. **Форматирование диалога**  
        Всю передачу оформить как череду реплик двух ведущих, пример оформления ниже:
        [WOMAN] Samantha: Приветствую вас в «Некрасовка daily»! [clears throat] [MAN] John: Добрый день, дорогие слушатели! Сегодня… [laughs]

        – переключайтесь между спикерами по необходимости.

        6. **Теги suno bark‑ai**  
        Обязательно использовать:
        - `[laughter]`, `[laughs]`, `[clears throat]`  
        - Дефис–тире (`—`) для естественных пауз  
        - `— or …` для заминок речи  
        - `♪ … ♪` для песенных фрагментов  
        - `CAPITALIZATION` для ударения слов
        d 
        7. Добавь 2-3 раза в некоторые моменты передачи тег [interrupt-podcast], на это место будет вставлена музыка-перебивка (как в радио эфирах)

        # Ожидаемый результат
        Готовый текстовый сценарий радиопередачи «Некрасовка daily» с чётким разделением на секции, спикерами и эффектами — полностью готовый к передаче на вход Bark TTS.

        {{DATE}}  = ищи дату в тексте
        {{NEWSPAPER_TEXT}} = {magazine_text}
        '''
        prompt = {
            "modelUri": f"gpt://{catalog_id}/yandexgpt/rc",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,
                "maxTokens": "100000"
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt_text
                }
            ]
        }


        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {API_KEY}"
        }
        # Адрес для обращения к модели 

        url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'

        data = {}

        # Указываем тип модели
        #'gpt://<идентификатор_каталога>/yandexgpt-lite'
        data['modelUri'] = f"gpt://b1g9016l80j9p7jb12rq/llama/rc"

        # Настраиваем дополнительные параметры модели
        data['completionOptions'] = {'stream': False,
                                    'temperature': 0.3,
                                    'maxTokens': 100000}

        # Указываем контекст для модели
        data['messages'] =  [
                {
                    "role": "user",
                    "text": prompt_text
                }
            ]
        print(data)
        # Получаем ответ модели
        response = requests.post(url, headers={'Authorization': 'Bearer ' + token}, json = data).json()
        return response['result']['alternatives'][0]['message']['text']

# --- LLM Factory ---
class LLMFactory:
    @staticmethod
    def get_formatter(model_name: str) -> LLMFormatter:
        model_name_lower = model_name.lower()
        if model_name_lower == "yandexgpt":
            return YandexGPTFormatter()
        elif model_name_lower == "llama":
            return LlamaFormatter()
        else:
            # Default or error
            print(f"Warning: Unknown model '{model_name}'. Defaulting to YandexGPT.")
            # raise HTTPException(status_code=400, detail=f"Unsupported LLM model: {model_name}")
            return YandexGPTFormatter() # Or return a default

# --- API Endpoint ---
class FormatRequest(BaseModel):
    text: str
    model: str # Expects "yandexgpt" or "llama" (case-insensitive)

@app.post("/format")
async def format_text(request: FormatRequest):
    """Accepts text and a model name, returns formatted text."""
    try:
        formatter = LLMFactory.get_formatter(request.model)
        print(f"LLM Service: Received text (length {len(request.text)}), model: {request.model}")
        formatted_text = await formatter.format(request.text)
        print(f"LLM Service: Formatted text length {len(formatted_text)}")
        return {"formatted_text": formatted_text}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"LLM Service Error formatting text: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during formatting: {e}")

if __name__ == "__main__":
    # Ensure you run this on a different port, e.g., 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)