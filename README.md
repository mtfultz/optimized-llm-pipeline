## 🔧 Quick start

```bash
git clone https://github.com/mtfultz/optimized-llm-pipeline && cd optimized-llm-pipeline
cp .env.example .env        
docker compose up -d        # vLLM + FastAPI + GUI
open http://localhost:8080  # GUI
```
| Metric                | RTX 4090     |
| --------------------- | ------------ |
| Latency (P50, 32 tok) | **190 ms**   |
| Throughput            | **42 tok/s** |
| VRAM                  | **15.8 GB**  |

Live tunnel: https://yearly-july-watson-maintaining.trycloudflare.com/


