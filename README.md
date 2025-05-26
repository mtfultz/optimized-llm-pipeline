<!-- 🚀 QUICK STATUS BADGES -->
[![CI](https://github.com/mtfultz/optimized-llm-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/mtfultz/optimized-llm-pipeline/actions/workflows/ci.yml)
![MIT License](https://img.shields.io/github/license/mtfultz/optimized-llm-pipeline)
![Docker (image size)](https://img.shields.io/docker/image-size/library/python/3.11-slim?label=api%20image)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Last commit](https://img.shields.io/github/last-commit/mtfultz/optimized-llm-pipeline)


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

![image](https://github.com/user-attachments/assets/903bb9f4-6490-410d-a955-5e36ffc49b06)

Example Usage:
“Summarise in two sentences the Heat-Affected-Zone Review findings for penetration PI3480864.”
“List every interim compensatory measure proposed for Dresden’s Turbine Building in the 1984 Appendix R exemption request.”
“Compare the fire-detection coverage on Side 1 versus Side 2 for penetration PI3350820.”
