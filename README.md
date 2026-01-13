# Tesseract OCR Benchmark POC

Proof of Concept to evaluate the performance of Tesseract OCR in resource-constrained environments (Kubernetes pods).

### Objective

To determine if the observed resource issues were caused by the implementation/configuration or if Tesseract is inherently too demanding for our environment.

## Project Structure

```
tesseract-benchmark/
├── docker/
│   ├── pytesseract/
│   │   ├── Dockerfile
│   │   └── benchmark.py
│   └── native/
│       ├── Dockerfile
│       └── benchmark.sh
├── k8s/
│   ├── kind-config.yaml
│   ├── jobs-pytesseract.yaml
│   └── jobs-native.yaml
├── scripts/
│   ├── run_benchmark.sh
│   └── analyze_results.py
├── samples/
└── results/
```