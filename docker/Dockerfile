FROM milvusbootcamp/python:3 
WORKDIR /app/src
COPY . /app
RUN pip3 install -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
CMD python3 main.py
