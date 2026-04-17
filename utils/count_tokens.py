#!/usr/bin/env python3
"""
统计 SFT 训练数据 JSONL 文件的 token 数量：
- 总 token 数（所有消息）
- Assistant token 数（真正计算损失的部分）

用法:
    python count_tokens.py <file.jsonl> [--tokenizer <name>]

依赖:
    pip install transformers tiktoken tqdm
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────
# Tokenizer 加载
# ──────────────────────────────────────────────

def load_tokenizer(name: str):
    """
    按优先级尝试加载 tokenizer：
    1. tiktoken（openai 风格，速度快）
    2. HuggingFace transformers AutoTokenizer
    """
    if name in ("cl100k_base", "o200k_base", "p50k_base"):
        try:
            import tiktoken
            enc = tiktoken.get_encoding(name)
            def tokenize(text: str) -> int:
                return len(enc.encode(text))
            print(f"[tokenizer] 使用 tiktoken: {name}")
            return tokenize
        except ImportError:
            print("[警告] tiktoken 未安装，尝试 transformers ...", file=sys.stderr)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        def tokenize(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))
        print(f"[tokenizer] 使用 transformers: {name}")
        return tokenize
    except Exception as e:
        print(f"[错误] 加载 tokenizer 失败: {e}", file=sys.stderr)
        sys.exit(1)


# ──────────────────────────────────────────────
# 消息解析
# ──────────────────────────────────────────────

def extract_messages(record: dict) -> list[dict]:
    """
    支持常见的 SFT 数据格式：
    1. {"messages": [...]}                         ← OpenAI / Anthropic 标准格式
    2. {"conversations": [...]}                    ← ShareGPT 风格
    3. {"instruction": "...", "output": "..."}     ← Alpaca 风格（自动转换）
    4. {"prompt": "...", "response": "..."}        ← 简单问答
    """
    if "messages" in record:
        return record["messages"]

    if "conversations" in record:
        # ShareGPT: human/gpt role 映射
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        return [
            {"role": role_map.get(m.get("from", ""), m.get("from", "")),
             "content": m.get("value", "")}
            for m in record["conversations"]
        ]

    if "instruction" in record:
        msgs = []
        if record.get("system"):
            msgs.append({"role": "system", "content": record["system"]})
        user_content = record["instruction"]
        if record.get("input"):
            user_content += "\n" + record["input"]
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": record.get("output", "")})
        return msgs

    if "prompt" in record and "response" in record:
        return [
            {"role": "user",    "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]

    return []


# ──────────────────────────────────────────────
# 主统计逻辑
# ──────────────────────────────────────────────

def count_file(path: str, tokenize, show_per_sample: bool = False):
    total_tokens     = 0
    assistant_tokens = 0
    total_samples    = 0
    skipped          = 0

    # 角色名称集合（assistant）
    ASSISTANT_ROLES = {"assistant", "gpt", "model", "bot"}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"\n共 {len(lines)} 行，开始统计...\n")

    for i, line in enumerate(tqdm(lines, desc="处理中"), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[警告] 第 {i} 行 JSON 解析失败: {e}", file=sys.stderr)
            skipped += 1
            continue

        messages = extract_messages(record)
        if not messages:
            skipped += 1
            continue

        sample_total     = 0
        sample_assistant = 0

        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")

            # content 可能是列表（多模态），取文本部分
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )

            n = tokenize(str(content)) if content else 0
            sample_total += n

            if role.lower() in ASSISTANT_ROLES:
                sample_assistant += n

        total_tokens     += sample_total
        assistant_tokens += sample_assistant
        total_samples    += 1

        if show_per_sample:
            print(f"  样本 {i:>5}: 总={sample_total:>6}  assistant={sample_assistant:>6}")

    return total_samples, skipped, total_tokens, assistant_tokens


# ──────────────────────────────────────────────
# 报告输出
# ──────────────────────────────────────────────

def print_report(path, tokenizer_name, total_samples, skipped,
                 total_tokens, assistant_tokens):
    other_tokens = total_tokens - assistant_tokens
    ratio = assistant_tokens / total_tokens * 100 if total_tokens else 0

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  📄 文件        : {Path(path).name}")
    print(f"  🔤 Tokenizer   : {tokenizer_name}")
    print(sep)
    print(f"  有效样本数     : {total_samples:>12,}")
    print(f"  跳过样本数     : {skipped:>12,}")
    print(sep)
    print(f"  总 token 数    : {total_tokens:>12,}")
    print(f"  Assistant token: {assistant_tokens:>12,}  ({ratio:.1f}%)")
    print(f"  其他 token     : {other_tokens:>12,}  ({100-ratio:.1f}%)")
    print(sep)
    print(f"  平均总 token/样本       : {total_tokens/max(total_samples,1):>8.1f}")
    print(f"  平均 assistant token/样本: {assistant_tokens/max(total_samples,1):>8.1f}")
    print(sep)


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="统计 SFT JSONL 数据集的总 token 及 assistant token"
    )
    parser.add_argument("file", help="输入的 .jsonl 文件路径")
    parser.add_argument(
        "--tokenizer", "-t",
        default="cl100k_base",
        help="Tokenizer 名称（默认: cl100k_base）。\n"
             "可选: o200k_base, p50k_base, 或任意 HuggingFace 模型名，\n"
             "例如 Qwen/Qwen2-7B、meta-llama/Meta-Llama-3-8B 等",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="逐条打印每个样本的 token 数"
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"[错误] 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(1)

    tokenize = load_tokenizer(args.tokenizer)
    total_samples, skipped, total_tokens, assistant_tokens = count_file(
        args.file, tokenize, show_per_sample=args.verbose
    )
    print_report(
        args.file, args.tokenizer,
        total_samples, skipped,
        total_tokens, assistant_tokens
    )


if __name__ == "__main__":
    main()