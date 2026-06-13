# RLLM — Исследование vLLM: из чего сделан и что перенимаем

## Миссия

Переписать vLLM под Windows без Docker, используя:
- Python 3.10 (E:/Python3.10)
- PyTorch 2.71 (E:\3\R\VLLM)
- CUDA 11.8 (оптимально для NVIDIA T600)
- Rust — для ускорения критических частей (по аналогии с uv)

---

## Что такое vLLM

vLLM — движок инференса и сервинга LLM с высокой пропускной способностью.
- GitHub: https://github.com/vllm-project/vllm
- **82.8K звёзд**
- Лицензия: Apache-2.0

---

## Стек vLLM (из кода)

### Языки (по порядку объёма):
1. **Python 84.3%** — основная логика, модели, API
2. **CUDA 5.4%** — GPU-ядро (csrc/)
3. **Rust 5.1%** — сервер, токенизатор, парсеры (rust/)
4. **C++ 3.8%** — CMake, вспомогательные модули

### Python-зависимости (pyproject.toml):
- `torch>=2.4.0,<2.11.0` — PyTorch (НАША ЦЕЛЬ: 2.71)
- `setuptools-rust>=1.9.0` — сборка Rust из Python
- `cmake>=3.26.1` — сборка C++/CUDA
- `ninja` — ускорение сборки

### Rust-компоненты (rust/Cargo.toml):
- **Rust toolchain: 1.95** (edition 2024)
- **Workspace из 12 крейтов:**
  - `vllm-server` — HTTP/gRPC сервер (axum + tonic)
  - `vllm-tokenizer` — токенизация (fasttokens, tiktoken-rs)
  - `vllm-chat` — чат-форматирование
  - `vllm-reasoning-parser` — парсинг рассуждений
  - `vllm-tool-parser` — парсинг вызовов инструментов
  - `vllm-text` — обработка текста
  - `vllm-metrics` — метрики (prometheus)
  - `vllm-llm` — обёртка над LLM
  - `vllm-engine-core-client` — клиент к движку
  - `vllm-managed-engine` — управляемый движок
  - `vllm-mock-engine` — тестовый движок
  - `vllm-cmd` — CLI

### CUDA-компоненты (csrc/):
- `attention/` — PagedAttention, FlashAttention
- `core/` — ядро вычислений
- `quantization/` — квантизация (FP8, INT4, INT8)
- `moe/` — Mixture of Experts
- `cutlass_extensions/` — оптимизированные GEMM

---

## Ключевые находки: параллели uv ↔ vLLM

### uv использует:
- Rust для всего (100% Rust)
- PubGrub для резолвера
- Глобальный кэш + hardlinks
- Параллельные HTTP-загрузки

### vLLM использует:
- Rust для сервера + токенизатора (5% кода)
- Python для моделей + логики (84%)
- CUDA для GPU-ускорения (5%)
- CMake для сборки C++/CUDA

### Что можно перенять из uv в RLLM:
1. **Rust-бинарник** вместо Python-стартера — быстрый запуск
2. **Глобальный кэш моделей** — hardlinks вместо копирования
3. **Параллельная загрузка** — скачивание весов модели
4. **PubGrub-подобный резолвер** — подбор оптимальной конфигурации

---

## Наша сборка: Windows + CUDA 11.8 + T600

### Проблема:
- Официальный vLLM поддерживает Linux (CUDA) и macOS
- Windows — только через vllm-windows форк или кастомную сборку
- T600 = Turing архитектура, CC 7.5, поддерживает CUDA 11.8

### Что нужно для сборки (ПРОВЕРЕНО):
1. **Python 3.10.10** — `E:\Python310` ✅
2. **PyTorch 2.7.1+cu118** — установлен ✅
3. **CUDA 11.8** — работает ✅
4. **NVIDIA T600** — видна ✅
5. **Rust toolchain 1.95** — нужна установка
6. **CMake + Ninja** — нужна установка
7. **MSVC** — Visual Studio Build Tools

### Среда разработки (подтверждено):
```
E:\Python310\python.exe
PyTorch: 2.7.1+cu118
CUDA: 11.8
GPU: NVIDIA T600
```

### Стратегия:
- Берём vllm-windows как основу (уже адаптирован под Windows)
- Заменяем Docker-сборку на нативную
- Собираем Rust-компоненты через `build_rust.sh` (адаптировать под Windows)
- Собираем CUDA через CMake с CUDA 11.8

---

## Структура RLLM

```
E:/3/R/MC/RLLM/
├── research/           # Исследования
│   ├── vllm-tech-stack.md
│   ├── cuda-compatibility.md
│   └── rust-integration.md
├── src/                # Исходники (пока пусто)
├── build/              # Сборка (пока пусто)
└── README.md
```

---

## Следующие шаги

1. Проверить CUDA 11.8 на T600
2. Проверить PyTorch 2.71 + CUDA 11.8 совместимость
3. Адаптировать build_rust.sh под Windows
4. Собрать Rust-компоненты
5. Собрать CUDA-компоненты
6. Собрать Python-пакет

---

## Источники

- https://github.com/vllm-project/vllm
- https://github.com/vllm-project/vllm/blob/main/pyproject.toml
- https://github.com/vllm-project/vllm/blob/main/rust/Cargo.toml
- E:\3\R\VLLM\vllm-windows\ (локальная копия)
