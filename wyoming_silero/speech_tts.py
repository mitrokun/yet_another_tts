import logging
import asyncio
import re
import os
import torch
import numpy as np
from num2words import num2words
import eng_to_ipa as ipa

log = logging.getLogger(__name__)

# Конфигурация модели
MODEL_URL = 'https://models.silero.ai/models/tts/ru/v5_1_ru.pt'
MODEL_FILENAME = 'v5_1_ru.pt'

# Параметры аудио
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SAMPLE_WIDTH = 2  # 16 bit
DEFAULT_CHANNELS = 1

class _EnglishToRussianNormalizer:
    """
    Класс, инкапсулирующий всю логику для преобразования
    английских слов в русское фонетическое представление.
    """
    SIMPLE_ENGLISH_TO_RUSSIAN = {
        'a': 'э', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
        'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'и', 'z': 'з'
    }

    ENGLISH_EXCEPTIONS = {
        "google": "гугл", "apple": "эпл", "microsoft": "майкрософт",
        "samsung": "самсунг", "toyota": "тойота", "volkswagen": "фольцваген",
        "coca": "кока", "cola": "кола", "pepsi": "пэпси", "whatsapp": "вотсап",
        "telegram": "телеграм", "youtube": "ютуб", "instagram": "инстаграм",
        "facebook": "фэйсбук", "twitter": "твиттер", "iphone": "айфон",
        "tesla": "тесла", "spacex": "спэйс икс", "amazon": "амазон",
        "python": "пайтон", "AI": "эй+ай", "api": "эйпиай",
        "IT": "+ай т+и", "Wi-Fi": "вай фай", "RTX ": "эрте+икс",
        "zigbee": "зигби", "mqtt": "эмкутити", "ha": "аш-а",
        "work": "ворк", "world": "ворлд", "bird": "бёрд",
        "girl": "гёрл", "burn": "бёрн", "her": "хёр",
        "early": "ёрли", "service": "сёрвис",
        "a": "э", "the": "зе", "of": "оф", "and": "энд", "for": "фо",
        "to": "ту", "in": "ин", "on": "он", "is": "из", "or": "ор",
        "knowledge": "ноуледж", "new": "нью",
        "video": "видео", "ru": "ру", "com": "ком",
        "hot": "хот", "https": "аштитипиэс", "http": "аштитипи",
    }

    IPA_TO_RUSSIAN_MAP = {
            "ˈ": "", "ˌ": "", "ː": "",
            "p": "п", "b": "б", "t": "т", "d": "д", "k": "к", "g": "г",
            "m": "м", "n": "н", "f": "ф", "v": "в", "s": "с", "z": "з",
            "h": "х", "l": "л", "r": "р", "w": "в", "j": "й",
            "ʃ": "ш", "ʒ": "ж",
            "tʃ": "ч", "ʧ": "ч",
            "dʒ": "дж", "ʤ": "дж",
            "ŋ": "нг", "θ": "с", "ð": "з",
            "i": "и", "ɪ": "и", "ɛ": "э", "æ": "э", "ɑ": "а", "ɔ": "о",
            "u": "у", "ʊ": "у", "ʌ": "а", "ə": "э",
            "ər": "эр", "ɚ": "эр",
            "eɪ": "эй", "aɪ": "ай", "ɔɪ": "ой", "aʊ": "ау", "oʊ": "оу",
            "ɪə": "иэ", "eə": "еэ", "ʊə": "уэ",
        }

    def __init__(self):
        self._max_ipa_key_len = max(len(key) for key in self.IPA_TO_RUSSIAN_MAP.keys())

    def _convert_ipa_to_russian(self, ipa_text: str) -> str:
        result = ""
        pos = 0
        while pos < len(ipa_text):
            found_match = False
            for length in range(self._max_ipa_key_len, 0, -1):
                chunk = ipa_text[pos:pos + length]
                if chunk in self.IPA_TO_RUSSIAN_MAP:
                    result += self.IPA_TO_RUSSIAN_MAP[chunk]
                    pos += length
                    found_match = True
                    break
            if not found_match:
                pos += 1
        return result

    def _transliterate_word(self, match):
        word_original = match.group(0)
        normalized_word = word_original.replace("’", "'")

        if word_original in self.ENGLISH_EXCEPTIONS:
            return self.ENGLISH_EXCEPTIONS[word_original]

        word_lower = word_original.lower()
        if word_lower in self.ENGLISH_EXCEPTIONS:
            return self.ENGLISH_EXCEPTIONS[word_lower]

        try:
            ipa_transcription = ipa.convert(word_lower)
            ipa_transcription = re.sub(r'[/]', '', ipa_transcription).strip()

            if '*' in ipa_transcription:
                raise ValueError("IPA conversion failed.")

            russian_phonetics = self._convert_ipa_to_russian(ipa_transcription)
            russian_phonetics = re.sub(r'йй', 'й', russian_phonetics)
            russian_phonetics = re.sub(r'([чшщждж])ь', r'\1', russian_phonetics)
            return russian_phonetics
        except Exception:
            return ''.join(self.SIMPLE_ENGLISH_TO_RUSSIAN.get(c, c) for c in word_lower)

    def normalize(self, text: str) -> str:
        return re.sub(r"\b[a-zA-Z]+(?:['’][a-zA-Z]+)*\b", self._transliterate_word, text)


class SpeechTTS:
    _emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002600-\U000026FF" u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF" u"\u200D" u"\uFE0F"
        "]+",
        flags=re.UNICODE
    )
    _chars_to_delete_for_translate = "=#$“”„«»<>*\"‘’‚‹›'/"
    _map_1_to_1_from_for_translate = "—–−\xa0"
    _map_1_to_1_to_for_translate   = "--- "
    _translation_table = str.maketrans(
        _map_1_to_1_from_for_translate,
        _map_1_to_1_to_for_translate,
        _chars_to_delete_for_translate
    )
    _FINAL_CLEANUP_PATTERN = re.compile(r'[^а-яА-ЯёЁ+?!., -]+')

    def __init__(self, data_dir: str = ".") -> None:
        log.debug(f"Initializing Silero TTS v5.1 ...")
        
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.channels = DEFAULT_CHANNELS
        self._lock = asyncio.Lock()
        self._eng_normalizer = _EnglishToRussianNormalizer()
        
        # Настройка Torch для CPU
        self.device = torch.device('cpu')
        torch.set_num_threads(4)

        # Путь к файлу модели
        self.local_file = os.path.join(data_dir, MODEL_FILENAME)

        try:
            if not os.path.isfile(self.local_file):
                log.info(f"Downloading model from {MODEL_URL} to {self.local_file}...")
                torch.hub.download_url_to_file(MODEL_URL, self.local_file)
            else:
                log.info(f"Using cached model at {self.local_file}")

            self.model = torch.package.PackageImporter(self.local_file).load_pickle("tts_models", "model")
            self.model.to(self.device)
            log.info("Silero V5.1 model loaded successfully.")

        except Exception as e:
            log.error(f"Failed to load Silero model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Silero TTS: {e}") from e

    def _normalize_english(self, text: str) -> str:
        return self._eng_normalizer.normalize(text)
        
    def _choose_percent_form(self, number_str: str) -> str:
        if '.' in number_str or ',' in number_str: return "процента"
        try:
            number = int(number_str)
            if 10 < number % 100 < 20: return "процентов"
            last_digit = number % 10
            if last_digit == 1: return "процент"
            if last_digit in [2, 3, 4]: return "процента"
            return "процентов"
        except (ValueError, OverflowError): return "процентов"

    def _normalize_percentages(self, text: str) -> str:
        def replace_match(match):
            number_str_clean = match.group(1).replace(',', '.')
            try: return f" {number_str_clean} {self._choose_percent_form(number_str_clean)} "
            except (ValueError, OverflowError): return f" {number_str_clean} процентов "
        processed_text = re.sub(r'(\d+([.,]\d+)?)\s*\%', replace_match, text)
        processed_text = processed_text.replace('%', ' процентов ')
        return processed_text

    def _normalize_special_chars(self, text: str) -> str:
        text = self._emoji_pattern.sub(r'', text)
        text = text.translate(self._translation_table)
        text = text.replace('…', '.')
        text = re.sub(r':(?!\d)', ',', text)
        text = re.sub(r'([a-zA-Zа-яА-ЯёЁ])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Zа-яА-ЯёЁ])', r'\1 \2', text)
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _normalize_plus_before_number(self, text: str) -> str:
        pattern = re.compile(r'\+(?=\d)')
        return pattern.sub(' плюс ', text)

    def _normalize_numbers(self, text: str) -> str:
        def replace_number(match):
            num_str = match.group(0).replace(',', '.')
            try:
                if '.' in num_str:
                    parts = num_str.split('.')
                    integer_part_str, fractional_part_str = parts[0], parts[1]
                    if not integer_part_str or not fractional_part_str:
                        valid_num_str = num_str.replace('.', '')
                        return num2words(int(valid_num_str), lang='ru') if valid_num_str.isdigit() else num_str
                    integer_part_val, fractional_part_val = int(integer_part_str), int(fractional_part_str)
                    fractional_len = len(fractional_part_str)
                    integer_words, fractional_words = num2words(integer_part_val, lang='ru'), num2words(fractional_part_val, lang='ru')
                    if fractional_len == 1: return f"{integer_words} и {fractional_words}"
                    if fractional_part_val % 10 == 1 and fractional_part_val % 100 != 11:
                        if fractional_words.endswith("один"): fractional_words = fractional_words[:-4] + "одна"
                    if fractional_part_val % 10 == 2 and fractional_part_val % 100 != 12:
                        if fractional_words.endswith("два"): fractional_words = fractional_words[:-3] + "две"
                    if fractional_len == 2: return f"{integer_words} и {fractional_words} сотых"
                    if fractional_len == 3: return f"{integer_words} и {fractional_words} тысячных"
                    return f"{integer_words} точка {fractional_words}"
                else: return num2words(int(num_str), lang='ru')
            except (ValueError, OverflowError) as e:
                log.warning(f"Could not normalize number '{num_str}': {e}")
                return num_str
        return re.sub(r'\b\d+([.,]\d+)?\b', replace_number, text)

    def _cleanup_final_text(self, text: str) -> str:
        return self._FINAL_CLEANUP_PATTERN.sub(' ', text)

    def _synthesize_thread(self, text: str, speaker_name: str, sample_rate: int) -> bytes:
        """
        Запускается в отдельном потоке.
        """
        # apply_tts возвращает Tensor (float32)
        audio_tensor = self.model.apply_tts(
            text=text,
            speaker=speaker_name,
            sample_rate=sample_rate
        )
        
        audio_int16 = (audio_tensor * 32767).numpy().astype(np.int16)
        return audio_int16.tobytes()

    async def synthesize(self, text: str, speaker_name: str, speech_rate: float = 1.0) -> bytes | None:
        log.debug(f"Requested TTS. Speaker: {speaker_name}, Text: [{text[:50]}...]")
        
        normalized_text = self._normalize_percentages(text)
        normalized_text = self._normalize_special_chars(normalized_text)
        normalized_text = self._normalize_plus_before_number(normalized_text)
        normalized_text = self._normalize_numbers(normalized_text)
        normalized_text = self._normalize_english(normalized_text)
        normalized_text = self._cleanup_final_text(normalized_text)
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()

        if not normalized_text:
            return None

        # Обработка тега SSML для скорости, если поддерживается
        if speech_rate != 1.0:
            # Для V5.1 можно попробовать обернуть в тег prosody, 
            # но простой apply_tts может его не распарсить корректно без явного флага ssml=True (если он есть в API).
            # В документации Standalone apply_tts прост. Оставим пока без изменения скорости.
            pass

        try:
            async with self._lock:
                audio_bytes = await asyncio.to_thread(
                    self._synthesize_thread, 
                    normalized_text, 
                    speaker_name, 
                    self.sample_rate
                )
            
            return audio_bytes
        except Exception as e:
            log.error(f"Silero TTS synthesis failed: {e}", exc_info=True)
            return None