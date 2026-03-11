# Vector Store

Векторное хранилище для хранения и поиска эмбеддингов документов.
Архитектура основана на GoF паттернах проектирования.

## Архитектурные паттерны

### GoF Patterns
- **Strategy** — стратегии индексации, поиска и персистентности
- **Facade** — `VectorStoreImpl` упрощает работу с подсистемой
- **Factory** — `create_vector_store` для создания конфигураций

### GRASP Principles
- **Information Expert** — `VectorDocument` хранит вектор и метаданные
- **Low Coupling** — слабая связанность через интерфейсы
- **High Cohesion** — каждый класс отвечает за одну задачу
- **Controller** — `VectorStoreImpl` управляет потоком операций

## Установка

```bash
pip install faiss-cpu  # Для FAISS индекса
pip install numpy      # Для векторных операций
```

## Быстрый старт

```python
from vector_store import VectorStoreImpl, VectorDocument

# Создание хранилища
store = VectorStoreImpl()

# Добавление документов
docs = [
    VectorDocument(
        id="1",
        vector=[0.1, 0.2, 0.3],
        content="Первый документ",
        metadata={"source": "file.pdf"}
    ),
    VectorDocument(
        id="2", 
        vector=[0.9, 0.8, 0.7],
        content="Второй документ",
    ),
]
store.add_documents(docs)

# Поиск
results = store.search(query_vector=[0.15, 0.25, 0.35], top_k=2)
for result in results:
    print(f"ID: {result.document.id}, Score: {result.score}")
    print(f"Content: {result.document.content}")
```

## Фабрика для создания хранилищ

```python
from vector_store import create_vector_store, DistanceMetric

# In-memory хранилище (для небольших датасетов)
store = create_vector_store("memory")

# FAISS хранилище (для больших датасетов)
store = create_vector_store("faiss", dimension=384)

# С метрикой расстояния
store = create_vector_store(
    "memory",
    metric=DistanceMetric.DOT_PRODUCT
)
```

## Стратегии поиска

### Similarity Search (по умолчанию)

```python
from vector_store import SimilaritySearch, VectorStoreImpl

store = VectorStoreImpl(
    search_strategy=SimilaritySearch(normalize_scores=True)
)

results = store.search(query_vector, top_k=5)
```

### MMR (Maximal Marginal Relevance)

MMR балансирует между релевантностью и разнообразием результатов.

```python
from vector_store import MMRSearch, VectorStoreImpl

# lambda_param: 1 = только релевантность, 0 = только разнообразие
store = VectorStoreImpl(
    search_strategy=MMRSearch(lambda_param=0.7)
)

results = store.search(query_vector, top_k=5)
```

### Смена стратегии на лету

```python
from vector_store import MMRSearch

# Переключение на MMR
store.set_search_strategy(MMRSearch(lambda_param=0.5))

# Поиск с разнообразными результатами
results = store.search(query_vector, top_k=5)
```

## Индексы

### InMemoryIndex

Брутфорс поиск, подходит для небольших датасетов (< 10k векторов).

```python
from vector_store import InMemoryIndex, VectorStoreImpl, DistanceMetric

index = InMemoryIndex(metric=DistanceMetric.COSINE)
store = VectorStoreImpl(index=index)
```

### FaissIndex

Быстрый поиск на основе FAISS (Facebook AI Similarity Search).

```python
from vector_store import FaissIndex, VectorStoreImpl, DistanceMetric

index = FaissIndex(dimension=384, metric=DistanceMetric.COSINE)
store = VectorStoreImpl(index=index)
```

## Персистентность

### JSON (по умолчанию)

```python
# Сохранение
store.save("vector_store.json")

# Загрузка
from vector_store import VectorStoreImpl
store = VectorStoreImpl.load("vector_store.json")
```

### Pickle (быстрее)

```python
from vector_store import VectorStoreImpl, PicklePersistence

store = VectorStoreImpl(persistence=PicklePersistence())
store.save("vector_store.pkl")

loaded = VectorStoreImpl.load("vector_store.pkl")
```

## Интеграция с Document Processor

```python
from document_processor import DocumentVectorizer
from vector_store import VectorStoreImpl, create_vector_store

# Создание векторизатора
vectorizer = DocumentVectorizer(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Создание хранилища
store = create_vector_store("faiss", dimension=384)

# Обработка документов и добавление в хранилище
docs = vectorizer.vectorize_batch(["file1.pdf", "file2.docx"])

for doc in docs:
    for vector in doc.vectors:
        store.add_documents([
            VectorDocument(
                id=f"{doc.path}_{vector.chunk.metadata['chunk_index']}",
                vector=vector.values,
                content=vector.chunk.content,
                metadata=vector.chunk.metadata,
            )
        ])

# Поиск по тексту
from document_processor.core import Chunk

query = "искусственный интеллект"
chunk = Chunk(content=query)
query_vector = vectorizer.embedder.embed([chunk])[0].values

results = store.search(query_vector, top_k=5)
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.document.content}")
```

## Поиск с фильтром

```python
# Фильтр по метаданным
def filter_by_source(doc):
    return doc.metadata.get("source") == "important.pdf"

results = store.search(
    query_vector,
    top_k=5,
    filter_fn=filter_by_source
)
```

## API

### VectorStoreImpl

| Метод | Описание |
|-------|----------|
| `add_documents(docs)` | Добавить документы |
| `add_texts(texts, vectors, metadatas, ids)` | Добавить тексты с векторами |
| `search(query_vector, top_k, filter_fn)` | Поиск похожих документов |
| `search_by_text(query_text, embedder, top_k)` | Поиск по тексту (требует embedder) |
| `delete(doc_ids)` | Удалить документы по ID |
| `get(doc_id)` | Получить документ по ID |
| `get_all()` | Получить все документы |
| `clear()` | Очистить хранилище |
| `save(path)` | Сохранить на диск |
| `load(path)` | Загрузить с диска |
| `set_search_strategy(strategy)` | Изменить стратегию поиска |

### VectorDocument

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `id` | str | Уникальный идентификатор |
| `vector` | List[float] | Векторное представление |
| `content` | str | Текстовое содержимое |
| `metadata` | dict | Метаданные |
| `dimension` | int | Размерность вектора (property) |

### SearchResult

| Атрибут | Тип | Описание |
|---------|-----|----------|
| `document` | VectorDocument | Найденный документ |
| `score` | float | Оценка релевантности |
| `rank` | int | Позиция в результатах |

## Структура проекта

```
vector_store/
├── core/
│   ├── interfaces.py      # Абстракции и интерфейсы
│   └── __init__.py
├── indexes/
│   ├── in_memory_index.py # In-memory индекс
│   ├── faiss_index.py     # FAISS индекс
│   └── __init__.py
├── strategies/
│   ├── similarity_search.py # Поиск по схожести
│   ├── mmr_search.py        # MMR поиск
│   └── __init__.py
├── persistence/
│   ├── json_persistence.py  # JSON/Pickle персистентность
│   └── __init__.py
├── store.py                 # Основная реализация
└── __init__.py
```

## Запуск тестов

```bash
pytest tests/vector_store/ -v
pytest tests/vector_store/ -v --cov=vector_store
```

## Лицензия

MIT
