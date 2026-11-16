# Мультиагентная система коллективных дебатов LLM (пилотная версия)

## Установка

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
```

Для локального запуска моделей установите Ollama и загрузите три модели разных
разработчиков (`ollama pull <модель>`), затем впишите их имена в `config.example.json`
вместо `MODEL_A`, `MODEL_B`, `MODEL_C`.

## Запуск

```
python cli.py ask "Вопрос"                       # один сеанс дебатов
python cli.py eval data.jsonl --details out.csv  # сравнение методов на наборе вопросов
streamlit run app.py                             # веб-интерфейс
python -m pytest                                 # тесты (модели не требуются)
```

Формат набора данных — JSONL, по одному вопросу в строке:
`{"question": "...", "answers": ["эталон", "допустимый вариант"]}`.

## Состав

| Файл | Назначение |
| --- | --- |
| `debate/schemas.py` | конфигурация, сообщение, состояние графа |
| `debate/prompts.py` | ролевые инструкции и шаблоны запросов |
| `debate/parsing.py` | разбор ответа модели, нормализация |
| `debate/aggregation.py` | голосование, доля согласия |
| `debate/graph.py` | граф LangGraph: propose, debate, aggregate |
| `debate/baselines.py` | одиночный запрос, CoT, самосогласованность |
| `debate/storage.py` | протоколы в SQLite |
| `debate/evaluate.py` | точность, бутстреп, критерий Мак-Немара |
| `cli.py`, `app.py` | командная строка и веб-интерфейс |
| `tests/` | модульные тесты с имитатором модели |

## Что не проверено

Тесты выполняются на имитаторе модели. Работа с реальными моделями через Ollama
и веб-интерфейс с реальными моделями не запускались: проверьте их на своей машине.
