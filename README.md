1. open terminal
2. python -m venv env
3. env\Scripts\activate
4. pip install -r requirements.txt
5. py manage.py runserver


docker build --tag ai-project .
docker run --publish 8000:8000 ai-project