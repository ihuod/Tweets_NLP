# Twitter Analysis Project

## Настройка окружения

### 1. Активация виртуального окружения

```bash
# Активация виртуального окружения
source venv/bin/activate

# Или используйте скрипт запуска
./run.sh
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Запуск Jupyter Notebook

```bash
# После активации виртуального окружения
jupyter notebook
```

### 4. Настройка VS Code

1. Откройте проект в VS Code
2. Нажмите `Cmd+Shift+P` (Mac) или `Ctrl+Shift+P` (Windows)
3. Введите "Python: Select Interpreter"
4. Выберите интерпретатор из виртуального окружения: `./venv/bin/python`

### 5. Проверка окружения

В Jupyter Notebook или Python скрипте:

```python
import sys
print(sys.executable)  # Должен показать путь к Python в виртуальном окружении
```

## Структура проекта

```
tweets/
├── data/                  # Данные
│   ├── raw/              # Исходные данные
│   └── processed/        # Обработанные данные
├── notebooks/            # Jupyter ноутбуки
├── src/                  # Исходный код
│   ├── data/            # Модули для работы с данными
│   ├── features/        # Модули для создания признаков
│   ├── models/          # Модули с моделями
│   └── utils/           # Вспомогательные функции
└── tests/               # Тесты
```

## Использование

1. Активировать виртуальное окружение
2. Запустить Jupyter Notebook:
```bash
jupyter notebook
```
3. Открыть `notebooks/main.ipynb` 