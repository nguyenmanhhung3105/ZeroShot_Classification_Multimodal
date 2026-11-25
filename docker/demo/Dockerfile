FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
# Only upload validation needs Pillow. Kaggle CLI and model runtime stay outside this image.
RUN pip install --no-cache-dir 'pillow>=10,<14' \
    && useradd --uid 10001 --create-home demo
COPY src/vlm_project/__init__.py src/vlm_project/demo_service.py /app/src/vlm_project/
COPY src/vlm_project/demo_web/ /app/src/vlm_project/demo_web/
USER 10001:10001
EXPOSE 8765
CMD ["python", "-c", "from vlm_project.demo_service import serve; serve(web_host='0.0.0.0')"]
