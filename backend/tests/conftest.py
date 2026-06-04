import sys
from pathlib import Path


# 测试从项目根目录运行时，Python 默认不知道 backend/app 在哪里。
# 这里把 backend 加到导入路径中，让测试可以直接写 from app.xxx import ...
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

