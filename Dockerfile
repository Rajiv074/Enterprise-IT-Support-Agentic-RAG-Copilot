##Python base image
FROM python:3.13-slim

#creating working directory 
WORKDIR /app

#configuring the env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

#copy the requirement
COPY requirement.txt .

#installing the requirement and copying the source code
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirement.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]