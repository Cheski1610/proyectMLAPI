FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY modelo_logist.pkl .

COPY main_1.py .

ENTRYPOINT [ "python", "main_1.py" ]