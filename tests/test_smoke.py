import os, requests, json, pytest

BASE = os.getenv("BASE_URL", "http://localhost:8080")
def test_chat_endpoint():
    r = requests.post(f"{BASE}/chat",
        json={"prompt":"2+2?","max_tokens":8})
    assert r.status_code == 200
    j = r.json()
    assert "choices" in j and j["choices"][0]["message"]["content"]
