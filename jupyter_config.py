import os
import sys

# Путь к виртуальному окружению
venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')

# Добавление путей в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(venv_path, 'lib', 'python3.13', 'site-packages'))

# Настройка переменных окружения
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
os.environ['VIRTUAL_ENV'] = venv_path 