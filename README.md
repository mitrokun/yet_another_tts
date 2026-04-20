### Зависимости
```
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install wyoming num2words eng_to_ipa regex numpy scipy
```
### Запуск (на 10208 порту)
```
python3 -m wyoming_silero
```
## [Быстрый старт](https://www.youtube.com/watch?v=HUGh9NhMNjg) с [uv](https://docs.astral.sh/uv/getting-started/installation/) 

```
git clone https://github.com/mitrokun/yet_another_tts.git
cd yet_another_tts  
uv run python -m wyoming_silero
```
или можно сохранить кэш внутри рабочего каталога
```
UV_CACHE_DIR=.uv_cache uv run python -m wyoming_silero
```
---
Обновленные версии модели возможно указать в `speech_tts.py`, а если требуется полностью сменить язык, тогда также придется вносить правки в main (список голосов и языковой код).
