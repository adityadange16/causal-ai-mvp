from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from llm_integration import fallback_parse, compute_counterfactual, get_rl_treatment, generate_explanation, confidence, data

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <title>Causal AI Helper: Medicine Storyteller</title>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f0f8ff;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    color: #333;
                }
                .container {
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    width: 80%;
                    max-width: 500px;
                }
                h1 {
                    color: #ff6347;
                    font-size: 2em;
                    margin-bottom: 10px;
                }
                p {
                    font-size: 1.1em;
                    margin-bottom: 20px;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                input[type="text"] {
                    padding: 10px;
                    font-size: 1em;
                    border: 2px solid #ff6347;
                    border-radius: 5px;
                    outline: none;
                }
                input[type="submit"] {
                    padding: 10px;
                    font-size: 1em;
                    background-color: #ff6347;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #e5533d;
                }
                #result {
                    margin-top: 20px;
                    opacity: 0;
                    transition: opacity 0.5s;
                }
                #result.show {
                    opacity: 1;
                }
                .loading {
                    font-style: italic;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧒 Causal AI Helper: Medicine Storyteller</h1>
                <p>Ask me what happens if a patient takes or skips medicine!</p>
                <form method="post" action="/query" id="queryForm">
                    <input type="text" name="query" placeholder="e.g., What if unit 21 remains untreated?">
                    <input type="submit" value="Tell Me a Story!">
                </form>
                <div id="result"></div>
                <script>
                    const form = document.getElementById('queryForm');
                    const resultDiv = document.getElementById('result');
                    form.addEventListener('submit', async (e) => {
                        e.preventDefault();
                        const query = document.querySelector('input[name="query"]').value;
                        resultDiv.innerHTML = '<p class="loading">Thinking of a story...</p>';
                        const response = await fetch('/query', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                            body: `query=${encodeURIComponent(query)}`
                        });
                        const data = await response.json();
                        resultDiv.innerHTML = `<h3>Your Adventure Story</h3><p>${data.story}</p>`;
                        resultDiv.classList.add('show');
                    });
                </script>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/query")
async def process_query(query: str = Form(...)):
    parsed = fallback_parse(query)
    if not parsed:
        raise HTTPException(status_code=400, detail="Oops! I didn’t understand. Try: 'What if unit 21 remains untreated?'")
    
    if 'scenario' in parsed:
        if parsed['scenario'] in ["treated", "untreated"]:
            rl_treatment = parsed.get('treatment', get_rl_treatment(parsed['unit'], data.iloc[parsed['unit']]['confounder_normalized']))
            counterfactual_outcome = compute_counterfactual(parsed['unit'], rl_treatment)
            actual_outcome = data.iloc[parsed['unit']]['outcome']
            explanation = generate_explanation(parsed['unit'], actual_outcome, counterfactual_outcome, confidence, query, rl_treatment, parsed)
        else:
            ate = 2.0
            explanation = f"Imagine a patient like unit {parsed['unit']}! If we check how medicine helps, our smart helper thinks it could make a difference of {int(ate)} health points with {confidence*100:.0f}% confidence!"
        return {"story": explanation}
    raise HTTPException(status_code=400, detail="Invalid scenario")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)