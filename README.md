# Document Processor

Модуль для конвертации PDF и DOCX документов в векторные представления (embeddings).
Архитектура основана на GoF и GRASP паттернах проектирования.

## Архитектурные паттерны

### GoF Patterns
- **Strategy** — стратегии чанкинга и embedding моделей
- **Factory Method** — фабрика загрузчиков документов
- **Facade** — упрощённый интерфейс через `DocumentVectorizer`
- **Adapter** — адаптеры для сторонних библиотек (PyMuPDF, python-docx)

### GRASP Principles
- **Information Expert** — доменная логика в моделях (Document, Chunk, Vector)
- **Low Coupling** — слабая связанность через интерфейсы
- **Controller** — `DefaultDocumentProcessor` управляет потоком обработки
- **High Cohesion** — каждый класс отвечает за одну задачу

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

```python
from document_processor import DocumentVectorizer

# Инициализация с настройками по умолчанию
vectorizer = DocumentVectorizer()

# Векторизация документа
doc = vectorizer.vectorize("document.pdf")
print(f"Создано векторов: {len(doc.vectors)}")
print(f"Размерность: {doc.vectors[0].dimension}")

# Пакетная обработка
docs = vectorizer.vectorize_batch(["file1.pdf", "file2.docx"])
```

## Конфигурация

### Настройки чанкинга

```python
from document_processor import DocumentVectorizer, RecursiveChunker

# Рекурсивный чанкер (рекомендуется)
vectorizer = DocumentVectorizer(
    chunk_size=500,      # Размер чанка в символах
    chunk_overlap=50,    # Перекрытие между чанками
)

# Фиксированный чанкер
from document_processor import FixedSizeChunker
chunker = FixedSizeChunker(chunk_size=1000, overlap=100)
vectorizer.set_chunker(chunker)
```

### Выбор embedding модели

```python
# Mock модель для тестов (быстро, не требует загрузки)
vectorizer = DocumentVectorizer(embedding_model="mock")

# Real модель (требует интернета для первой загрузки)
vectorizer = DocumentVectorizer(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Custom модель
from document_processor import SentenceTransformerEmbedder
embedder = SentenceTransformerEmbedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cuda"  # или "cpu"
)
vectorizer.set_embedder(embedder)
```

## API

### DocumentVectorizer

| Метод | Описание |
|-------|----------|
| `vectorize(path)` | Векторизовать один документ |
| `vectorize_batch(paths)` | Векторизовать несколько документов |
| `extract_text(path)` | Извлечь текст без векторизации |
| `set_chunker(chunker)` | Изменить стратегию чанкинга |
| `set_embedder(embedder)` | Изменить модель эмбеддингов |

### Document

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `path` | str | Путь к файлу |
| `type` | DocumentType | Тип документа (PDF/DOCX) |
| `content` | str | Полный текст документа |
| `chunks` | List[Chunk] | Список чанков |
| `vectors` | List[Vector] | Список векторов |
| `metadata` | dict | Метаданные документа |

### Chunk

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `content` | str | Текст чанка |
| `metadata` | dict | Метаданные (source, chunk_index, etc.) |

### Vector

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `values` | List[float] | Векторные значения |
| `chunk` | Chunk | Связанный чанк |
| `dimension` | int | Размерность вектора |

## Расширение

### Добавление нового загрузчика

```python
from document_processor.core import DocumentLoader, Document, DocumentType

class TXTLoader(DocumentLoader):
    def load(self, path: str) -> Document:
        with open(path) as f:
            content = f.read()
        return Document(path=path, type=DocumentType.TXT, content=content)
    
    def supports(self, document_type: DocumentType) -> bool:
        return document_type == DocumentType.TXT

# Регистрация
from document_processor.loaders import LoaderFactory
LoaderFactory.register_loader(DocumentType.TXT, TXTLoader)
```

### Добавление новой стратегии чанкинга

```python
from document_processor.core import ChunkingStrategy, Chunk

class SemanticChunker(ChunkingStrategy):
    def chunk(self, text: str, metadata: dict) -> list[Chunk]:
        # Ваша логика чанкинга
        return [Chunk(content=text, metadata=metadata)]
```

## Запуск тестов

```bash
pytest tests/ -v
pytest tests/ -v --cov=document_processor
```

## Структура проекта

```
document_processor/
├── core/
│   ├── interfaces.py      # Абстракции и интерфейсы
│   ├── processor.py       # Контроллер обработки
│   └── __init__.py
├── loaders/
│   ├── pdf_loader.py      # Загрузчик PDF
│   ├── document_loader.py # Загрузчик DOCX
│   ├── factory.py         # Фабрика загрузчиков
│   └── __init__.py
├── chunkers/
│   ├── fixed_chunker.py   # Фиксированный чанкинг
│   ├── recursive_chunker.py # Рекурсивный чанкинг
│   └── __init__.py
├── embeddings/
│   ├── sentence_transformer.py # Real embeddings
│   ├── mock_embedder.py        # Mock для тестов
│   └── __init__.py
├── adapters/              # Адаптеры (расширения)
├── facade.py              # Фасад DocumentVectorizer
└── __init__.py
```

## Лицензия

MIT
