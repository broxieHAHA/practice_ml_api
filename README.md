# Practice ML API — MLOps Stack

REST API для классификации ирисов с полным MLOps-стеком.

## Стек технологий

- **MLFlow** — трекинг экспериментов и реестр моделей
- **PostgreSQL** — хранение метаданных MLFlow
- **MinIO** — S3-совместимое хранилище артефактов модели
- **FastAPI** — REST API для инференса модели
- **Docker** — контейнеризация всех сервисов

## Структура проекта
practice_ml_api/
├── docker/
│   ├── Dockerfile            # FastAPI сервис
│   ├── Dockerfile.mlflow     # MLFlow сервер
│   └── Dockerfile.train      # Обучение модели
├── src/
│   ├── main.py               # FastAPI приложение
│   └── train.py              # Скрипт обучения модели
├── docker-compose.yml
├── requirements.txt
└── README.md

- Python 3.11

## Запуск проекта

### 1. Клонировать репозиторий

```bash
git clone https://github.com/broxieHAHA/practice_ml_api.git
cd practice_ml_api
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Поднять инфраструктуру

```bash
docker-compose up -d --build postgres minio minio-init mlflow
```

### 4. Обучить модель

```bash
docker-compose --profile train up --build trainer
```

### 5. Запустить FastAPI

```bash
docker-compose up -d --build fastapi
```

## Сервисы

| Сервис | URL | Описание |
|--------|-----|----------|
| FastAPI Swagger | http://localhost:8000/docs | REST API документация |
| MLFlow UI | http://localhost:5000 | Эксперименты и модели |
| MinIO Console | http://localhost:9001 | Хранилище артефактов |

MinIO логин: `minioadmin`, пароль: `minioadmin`

## Остановка

```bash
# Остановить все контейнеры
docker-compose down

# Остановить и удалить все данные
docker-compose down -v
```