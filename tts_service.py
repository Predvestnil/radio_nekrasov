# tts_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import uuid
import asyncio
from config import GENERATED_AUDIO_DIR, GENERATED_AUDIO_STATIC_PATH

app = FastAPI()


os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)

import nltk
import numpy as np
import re
import os
import glob
import random
import numpy as np
import torch
import torchaudio
from IPython.display import Audio
from transformers import AutoProcessor, BarkModel
from tqdm import tqdm

class TextToSpeechBark:
    def __init__(self, speaker_lookup, speaker_pattern, quiet_speakers):
        self.speaker_lookup = speaker_lookup
        self.speaker_pattern = speaker_pattern
        self.quiet_speakers = quiet_speakers
        # Пути к папкам с mp3
        self.intro_files = glob.glob(os.path.join("intro", "*.mp3"))
        self.interrupt_files = glob.glob(os.path.join("interrupt", "*.mp3"))
        self.device = "cuda"
        print(f"Using device: {self.device}")

        # Загрузка модели
        print("Loading Bark model...")
        model = BarkModel.from_pretrained(
            "suno/bark",
            torch_dtype=torch.float16,
        ).to(self.device)
        print("Model loaded successfully.")

        # Загрузка процессора
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        print("Processor loaded successfully.")

        # Оптимизация модели
        print("Optimizing model with BetterTransformer...")
        self.model = model.to_bettertransformer()
        print("Model optimized successfully.")

        # Получение частоты дискретизации
        self.sampling_rate = model.generation_config.sample_rate
        print(f"Sampling rate: {self.sampling_rate} Hz")

        # Настройка размера пакета и паузы
        self.BATCH_SIZE = 32
        self.silence = np.zeros(int(0.25 * self.sampling_rate))
        print(f"Batch size set to: {self.BATCH_SIZE}")
        print(f"Silence duration: {len(self.silence) / self.sampling_rate:.2f} seconds")

        # Готовность к генерации аудио
        print("Setup complete. Ready to generate audio.")
    
    def generate_speech(self, text):
        different_speakers, phrase_numbers, chunks = self._split_by_speakers(text)
        sequence = []

        for speaker, text in chunks:
            if speaker == '[interrupt-podcast]':
                sequence.append(('interrupt', None, None))
            else:
                sequence.append(('speech', speaker, text))
                different_speakers.setdefault(speaker, []).append(text)
        
        print(different_speakers)
        
        speakers_audio = self._generate_voices_in_batch(different_speakers)
        
        pieces = []
        audio_idx = {sp: 0 for sp in different_speakers}
        for evt_type, speaker, text in sequence:
            if evt_type == 'interrupt':
                path = random.choice(self.interrupt_files)
                wave, sr = torchaudio.load(path)
                if sr != self.sampling_rate:  # Убеждаемся в одинаковой частоте дискретизации
                    wave = torchaudio.functional.resample(wave, sr, self.sampling_rate)
                mono = wave.mean(dim=0).cpu().numpy()
                pieces.append(mono)
            elif evt_type == 'speech':
                idx = audio_idx[speaker]
                audio = speakers_audio[speaker][idx]
                pieces.append(audio)
                pieces.append(self.silence.copy())  # Добавляем тишину после речи
                audio_idx[speaker] += 1
                
        intro_path = random.choice(self.intro_files)
        intro_waveform, intro_sr = torchaudio.load(intro_path)  # [channels, time]

        # 2. Определяем целевую SR для фраз ведущих (из модели)
        target_sr = self.model.generation_config.sample_rate

        # 3. Ресемплинг интро для соответствия target_sr без изменения скорости
        # Чтобы сохранить оригинальную скорость, учитываем длительность
        if intro_sr != target_sr:
            intro_waveform = torchaudio.functional.resample(
                intro_waveform, orig_freq=intro_sr, new_freq=target_sr
            )

        # 4. Готовим new_audio (фразы ведущих) — уже в target_sr, преобразуем в тензор
        new_wf = torch.from_numpy(np.concatenate(pieces))
        if new_wf.ndim == 1:
            new_wf = new_wf.unsqueeze(0)  # [1, time]

        # 5. Выравниваем число каналов
        n_intro_ch = intro_waveform.shape[0]
        n_new_ch = new_wf.shape[0]

        if n_intro_ch > n_new_ch:
            # Дублируем моно-фразу в стерео/больше каналов
            new_wf = new_wf.repeat(n_intro_ch, 1)
        elif n_intro_ch < n_new_ch:
            # Обрезаем лишние каналы фразы
            new_wf = new_wf[:n_intro_ch, :]

        # 6. Склеиваем вдоль временной оси
        output_waveform = torch.cat([intro_waveform, new_wf], dim=1)  # [channels, total_time]

        # 7. Преобразуем в numpy и сохраняем
        output_np = output_waveform.cpu().numpy()
        return output_np

    def _split_by_speakers(self, text):
        text_prompt = text.replace("\n", " ").strip()

        # Разделяем текст по этому шаблону
        segments = self.speaker_pattern.split(text_prompt)[1:]  # Пропускаем возможный пустой первый элемент

        # Собираем список (спикер, текст)
        processed_segments = []
        for i in range(0, len(segments), 2):
            speaker_tag = segments[i].strip()
            text = segments[i+1].strip()
            processed_segments.append((speaker_tag, text))

        # Разбиваем на чанки длиной до 200 символов, вставляя музыку/прерывания как отдельные сегменты
        chunks = []
        for speaker, text in processed_segments:
            if speaker in ["[intro-music]", "[interrupt-podcast]"]:
                # Момент музыки или прерывания без текста
                chunks.append((speaker, ""))
                continue
            # Для обычных спикеров — разбиваем текст на предложения
            sents = nltk.sent_tokenize(text)
            current_chunk = ""
            for sent in sents:
                if len(current_chunk) + len(sent) < 200:
                    current_chunk += sent + " "
                else:
                    chunks.append((speaker, current_chunk.strip()))
                    current_chunk = sent + " "
            if current_chunk:
                chunks.append((speaker, current_chunk.strip()))

        # Сортируем как в изначальном коде (по имени спикера)
        sorted_chunks = sorted(enumerate(chunks), key=lambda x: x[1][0])

        # Собираем итоговый словарь
        different_speakers = {}
        phrase_numbers = {}
        for idx, (speaker, chunk_text) in sorted_chunks:
            if speaker not in different_speakers:
                different_speakers[speaker] = []
                phrase_numbers[speaker] = []
            different_speakers[speaker].append(chunk_text)
            phrase_numbers[speaker].append(idx)
        return different_speakers, phrase_numbers, chunks
        
    def _generate_voices_in_batch(self, different_speakers):
        speakers_audio = {}
        with torch.inference_mode():
            for speaker, phrases in tqdm(different_speakers.items(), desc="Generating per-speaker audio"):
                voice_preset = self.speaker_lookup.get(speaker, "v2/ru_speaker_3")
                speakers_audio[speaker] = []
                for i in range(0, len(phrases), self.BATCH_SIZE):
                    batch_texts = phrases[i : i + self.BATCH_SIZE]
                    inputs = self.processor(batch_texts, voice_preset=voice_preset)
                    inputs['history_prompt'] = inputs['history_prompt'].to(self.device)
                    speech_output, output_lengths = self.model.generate(
                        **inputs.to(self.device),
                        return_output_lengths=True,
                        min_eos_p=0.7,
                        fine_temperature=0.9,
                        coarse_temperature=0.4
                    )
                    if speaker in quiet_speakers:
                        speech_output *= 3
                    batch_waves = [
                        out[:length].cpu().numpy()
                        for out, length in zip(speech_output, output_lengths)
                    ]
                    speakers_audio[speaker].extend(batch_waves)  # Только аудио, без тишины
        return speakers_audio
    
quiet_speakers = ["[WOMAN] Samantha:"]
# Настройки голосов 
speaker_lookup = {
    "[WOMAN] Samantha:": "v2/ru_speaker_9",
    "[MAN] John:": "v2/ru_speaker_3"
}

# Расширенный шаблон для всех тегов и спикеров
speaker_pattern = re.compile(
    r'(\[intro-music\]|\[interrupt-podcast\]|\[WOMAN\] Samantha:|\[MAN\] John:)'
)

class SynthesizeRequest(BaseModel):
    text: str

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """Accepts text and returns the path to the generated audio file."""
    try:
        print(f"TTS Service: Received text length {len(request.text)}")
        if not request.text:
             raise HTTPException(status_code=400, detail="Cannot synthesize empty text")
        
        # Исходный текст
        text_prompt = request.text

        model = TextToSpeechBark(speaker_lookup, speaker_pattern, quiet_speakers)
        audio = model.generate_speech(text_prompt)

        audio_path = "output.wav"
        torchaudio.save(audio_path, torch.tensor(audio), model.sampling_rate)

        print(f"Сохранили output.wav, SR = {model.sampling_rate}, shape = {audio.shape}")
        return {"audio_path": audio_path}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"TTS Service Error synthesizing speech: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during TTS: {e}")

if __name__ == "__main__":
    # Ensure you run this on a different port, e.g., 8003
    uvicorn.run(app, host="0.0.0.0", port=8003)