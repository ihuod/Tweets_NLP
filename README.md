# NLP with Disaster Tweets

## Environment Setup

### 1. Activate Virtual Environment

```bash
# Activate the virtual environment
source venv/bin/activate

# Or use the launch script
./run.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
# After activating the virtual environment
jupyter notebook
```

### 4. VS Code Setup

1. Open the project in VS Code
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
3. Type "Python: Select Interpreter"
4. Select the interpreter from the virtual environment: `./venv/bin/python`

### 5. Verify the Environment

In a Jupyter Notebook or Python script:

```python
import sys
print(sys.executable)  # Should display the path to Python in the virtual environment
```

## Project Structure

```
tweets/
├── data/                 # Data
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering modules
│   ├── models/           # Model modules
│   └── utils/            # Utility functions
└── tests/                # Tests
```

## Usage

1. Activate the virtual environment
2. Start Jupyter Notebook:
```bash
jupyter notebook
```
3. Open `notebooks/main.ipynb`  
```
