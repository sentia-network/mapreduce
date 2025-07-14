FROM python:3.11.3

WORKDIR /app
COPY ./nltk_data /root/nltk_data
COPY pyproject.toml ./

RUN apt-get update && \
    apt-get install -y pandoc && \
    rm -rf /var/lib/apt/lists/* && \
    pip install poetry -i https://mirrors.aliyun.com/pypi/simple/ && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

RUN [ "python", "-c", "from langchain.text_splitter import RecursiveCharacterTextSplitter; RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)  # warm up" ]

COPY . .
RUN python -c "import compileall; compileall.compile_path(maxlevels=10)" && \
    python -m compileall .

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
