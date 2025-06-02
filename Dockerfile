# Python 3.10 베이스 이미지
FROM python:3.10

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
pip install -r /tmp/requirements.txt

# 작업 디렉토리 설정 (리눅스 스타일)
WORKDIR /workspace

# 기본 쉘
CMD ["bash"]
