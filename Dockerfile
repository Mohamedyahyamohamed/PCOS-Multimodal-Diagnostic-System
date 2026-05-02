# استخدام صورة بايثون خفيفة ومستقرة
FROM python:3.10-slim

# إعداد مسار العمل الأساسي
WORKDIR /code

# نسخ ملف المتطلبات وتثبيت المكتبات
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# نسخ باقي ملفات المشروع والنماذج
COPY . /code

# فتح المنفذ الافتراضي لـ Hugging Face Spaces
EXPOSE 7860

# تشغيل خادم FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]