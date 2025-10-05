from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>Hello! The server is working!</h1>
        </body>
    </html>
    """