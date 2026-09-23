#!/bin/bash
set -e

echo "Running Forced vs Expanded Comparison (Tiny)..."
python experiments/followups/forced_vs_expanded/forced_runner.py --tiny --output experiments/followups/forced_vs_expanded/forced_tiny.csv
python experiments/m3v2/m3v2_runner.py --tiny --output experiments/followups/forced_vs_expanded/expanded_tiny.csv

echo "Pulling models for Cross-Model Replication..."
# we pull lightweight versions to save time/space
ollama pull llama3:8b-instruct-q4_K_M || ollama pull llama3 || echo "Pull failed"
ollama pull gemma2:9b-instruct-q4_K_M || ollama pull gemma2 || echo "Pull failed"
ollama pull mistral:7b-instruct-q4_K_M || ollama pull mistral || echo "Pull failed"

echo "Running Cross-Model Replication (Tiny)..."
python experiments/m3v2/m3v2_runner.py --tiny --ollama_model qwen2.5:3b-instruct-q4_K_M --output experiments/followups/cross_model/qwen_tiny.csv
python experiments/m3v2/m3v2_runner.py --tiny --ollama_model llama3 --output experiments/followups/cross_model/llama_tiny.csv || echo "llama3 run failed"
python experiments/m3v2/m3v2_runner.py --tiny --ollama_model gemma2 --output experiments/followups/cross_model/gemma_tiny.csv || echo "gemma2 run failed"
python experiments/m3v2/m3v2_runner.py --tiny --ollama_model mistral --output experiments/followups/cross_model/mistral_tiny.csv || echo "mistral run failed"

echo "All follow-up experiments completed."
