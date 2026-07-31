modern-cnn-inference-lab/
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── database.py
│   └── settings.py
├── ml/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── artifacts/
│       ├── model.pt
│       ├── model.onnx
│       └── metadata.json
├── scripts/
│   ├── benchmark_model.py
│   ├── load_test.py
│   └── query_metrics.sql
├── tests/
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .env.example
└── README.md