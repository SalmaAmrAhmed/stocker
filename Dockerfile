FROM 3.13.0a4-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE $PORT

CMD bash -c "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"
