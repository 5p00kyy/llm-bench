#!/usr/bin/env python3
"""
Speed test for llama-server models.
Measures tok/s for prompt processing and generation at various context lengths.
Outputs JSON results to data/speed/
"""

import json
import time
import sys
import os
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone

SERVER_URL = os.environ.get("LLAMA_SERVER", "http://127.0.0.1:8080")
API_KEY = os.environ.get("LLAMA_API_KEY", "Smiledog69!")

# Standard test prompts at different lengths
# Short prompt for generation speed, long prompts for prompt processing speed
PROMPTS = {
    "short": "Write a brief explanation of how gravity works.",
    "medium": None,  # Generated at runtime to target ~2K tokens
    "long": None,    # Generated at runtime to target ~8K tokens
}

def make_medium_prompt():
    """~2K token prompt for medium context test"""
    base = "Analyze the following technical document and provide a detailed summary:\n\n"
    # Repeat filler to reach ~2K tokens
    filler = (
        "The system architecture consists of multiple interconnected components. "
        "Each component operates independently while maintaining state consistency "
        "through a distributed consensus protocol. The primary data store uses a "
        "B-tree indexed structure with write-ahead logging for durability. "
        "Replication occurs asynchronously with configurable consistency levels. "
    )
    return base + (filler * 40) + "\n\nProvide a 200-word summary of the above."

def make_long_prompt():
    """~8K token prompt for long context test"""
    base = "You are analyzing a large codebase. Here is the full source:\n\n"
    filler = (
        "def process_batch(items, config):\n"
        "    results = []\n"
        "    for item in items:\n"
        "        if item.status == 'pending':\n"
        "            result = transform(item, config.params)\n"
        "            results.append(result)\n"
        "    return aggregate(results)\n\n"
    )
    return base + (filler * 130) + "\nList all function names and their purposes."


def chat_completion(model, prompt, max_tokens=256, temperature=0.0):
    """Send a chat completion request and measure timing."""
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()

    req = urllib.request.Request(url, data=payload, method='POST')
    req.add_header('Authorization', f'Bearer {API_KEY}')
    req.add_header('Content-Type', 'application/json')

    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read().decode())
        elapsed = time.time() - start
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Extract timing from llama.cpp server response if available
    timings = data.get("timings", {})

    result = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_s": round(elapsed, 2),
        "tok_per_s_overall": round(completion_tokens / elapsed, 1) if elapsed > 0 else 0,
    }

    # llama.cpp provides detailed timings
    if timings:
        result["prompt_eval_tok_s"] = round(timings.get("prompt_per_second", 0), 1)
        result["generation_tok_s"] = round(timings.get("predicted_per_second", 0), 1)
        result["prompt_eval_ms"] = round(timings.get("prompt_ms", 0), 1)
        result["generation_ms"] = round(timings.get("predicted_ms", 0), 1)
    else:
        # Estimate from overall timing (less accurate)
        if completion_tokens > 0 and elapsed > 0:
            result["generation_tok_s"] = result["tok_per_s_overall"]

    return result


def get_loaded_model():
    """Check which model is currently loaded."""
    url = f"{SERVER_URL}/v1/models"
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {API_KEY}')
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        models = data.get("data", [])
        if models:
            return models[0].get("id", "unknown")
    except:
        pass
    return None


def get_vram_usage():
    """Get current VRAM usage via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            used, total = line.strip().split(', ')
            gpus.append({"used_mib": int(used), "total_mib": int(total)})
        return gpus
    except:
        return None


def run_speed_test(model_id, runs=3):
    """Run full speed test battery for a model."""
    print(f"\n{'='*60}")
    print(f"Speed test: {model_id}")
    print(f"{'='*60}")

    # Warmup
    print("  Warmup...", end=" ", flush=True)
    warmup = chat_completion(model_id, "Hello", max_tokens=10)
    if "error" in warmup:
        print(f"FAILED: {warmup['error']}")
        return None
    print("OK")

    vram = get_vram_usage()

    tests = {
        "short_prompt": {
            "prompt": PROMPTS["short"],
            "max_tokens": 256,
            "desc": "Short prompt (~20 tok) → 256 gen tokens"
        },
        "medium_prompt": {
            "prompt": make_medium_prompt(),
            "max_tokens": 256,
            "desc": "Medium prompt (~2K tok) → 256 gen tokens"
        },
        "long_prompt": {
            "prompt": make_long_prompt(),
            "max_tokens": 256,
            "desc": "Long prompt (~8K tok) → 256 gen tokens"
        },
        "long_generation": {
            "prompt": "Write a detailed 2000-word essay about the history of computing, from Charles Babbage to modern AI. Include specific dates and names.",
            "max_tokens": 1024,
            "desc": "Short prompt → 1024 gen tokens"
        },
    }

    results = {}
    for test_name, test_config in tests.items():
        print(f"  {test_config['desc']}...", flush=True)
        test_results = []
        for i in range(runs):
            r = chat_completion(model_id, test_config["prompt"], test_config["max_tokens"])
            if "error" not in r:
                test_results.append(r)
                gen_speed = r.get("generation_tok_s", r.get("tok_per_s_overall", 0))
                pp_speed = r.get("prompt_eval_tok_s", 0)
                print(f"    Run {i+1}: pp={pp_speed} tok/s, gen={gen_speed} tok/s, "
                      f"prompt={r['prompt_tokens']} tok, gen={r['completion_tokens']} tok")
            else:
                print(f"    Run {i+1}: ERROR - {r['error'][:100]}")

        if test_results:
            # Take best of runs (most representative when GPU is warm)
            best = max(test_results, key=lambda x: x.get("generation_tok_s", x.get("tok_per_s_overall", 0)))
            avg_gen = sum(r.get("generation_tok_s", r.get("tok_per_s_overall", 0)) for r in test_results) / len(test_results)
            avg_pp = sum(r.get("prompt_eval_tok_s", 0) for r in test_results) / len(test_results)

            results[test_name] = {
                "description": test_config["desc"],
                "runs": len(test_results),
                "best_generation_tok_s": best.get("generation_tok_s", best.get("tok_per_s_overall", 0)),
                "avg_generation_tok_s": round(avg_gen, 1),
                "best_prompt_eval_tok_s": best.get("prompt_eval_tok_s", 0),
                "avg_prompt_eval_tok_s": round(avg_pp, 1),
                "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in test_results) / len(test_results)),
                "avg_completion_tokens": round(sum(r["completion_tokens"] for r in test_results) / len(test_results)),
            }

    return {
        "model": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": SERVER_URL,
        "hardware": {
            "gpus": "2x RTX 5060 Ti 16GB",
            "ram": "60GB DDR4-2133 ECC",
            "backend": "llama.cpp",
        },
        "vram_usage": vram,
        "tests": results,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Speed Test")
    parser.add_argument("--model", help="Model ID to test (default: auto-detect)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test (default: 3)")
    parser.add_argument("--output", help="Output JSON path (default: auto)")
    args = parser.parse_args()

    model = args.model or get_loaded_model()
    if not model:
        print("ERROR: No model specified and couldn't detect loaded model")
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Server: {SERVER_URL}")
    print(f"Runs per test: {args.runs}")

    result = run_speed_test(model, runs=args.runs)
    if not result:
        print("Speed test failed")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {model}")
    print(f"{'='*60}")
    for test_name, data in result["tests"].items():
        print(f"  {data['description']}")
        print(f"    Gen: {data['best_generation_tok_s']} tok/s (best), {data['avg_generation_tok_s']} tok/s (avg)")
        if data['best_prompt_eval_tok_s'] > 0:
            print(f"    PP:  {data['best_prompt_eval_tok_s']} tok/s (best), {data['avg_prompt_eval_tok_s']} tok/s (avg)")

    # Save
    if args.output:
        outpath = args.output
    else:
        slug = model.replace("/", "-").replace(" ", "-").lower()
        os.makedirs("data/speed", exist_ok=True)
        outpath = f"data/speed/{slug}.json"

    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
