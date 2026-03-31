import asyncio, json, logging, os
from dataclasses import dataclass, field
from pathlib import Path
import tiktoken, yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
API_URL       = os.getenv("OPENAI_BASE_URL", "")
API_KEY       = os.getenv("OPENAI_API_KEY", "")
MODEL         = os.getenv("MODEL_NAME", "Qwen3-30B-A3B-Thinking-2507")
SIM_MODEL     = os.getenv("SIM_MODEL", "") or MODEL
ORACLE_MODEL  = os.getenv("ORACLE_MODEL", "") or MODEL
DATA_DIR  = ROOT / "data"
OUT_DIR   = ROOT / "output"
CONV_DIR  = OUT_DIR / "conversations"
SFT_DIR   = OUT_DIR / "sft"
ENC       = tiktoken.get_encoding("cl100k_base")
@dataclass
class Persona:
    id: str
    attributes: dict = field(default_factory=dict)
    summary: str = ""
    fingerprint: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    selected_prompts: list = field(default_factory=list)
    def domain(self)    -> list: return self.fingerprint.get("domain", ["general"])
    def register(self)  -> str:  return self.fingerprint.get("register", "casual")
    def expertise(self) -> str:  return self.fingerprint.get("expertise_level", "mid")
    @property
    def refined_summary(self) -> str:
        return self.metadata.get("refined_summary", "") or self.summary

    @property
    def behavioral_metadata(self) -> dict:
        return self.metadata.get("behavioral_metadata", {})

    def to_block(self) -> str:
        """XML-tagged profile block for prompt injection."""
        # Prefer refined_summary if available
        summary = self.refined_summary or self.summary
        parts = [f"<summary>\n{summary}\n</summary>"]
        bm = self.behavioral_metadata
        if bm:
            parts.append(f"<behavioral_metadata>\n{json.dumps(bm, indent=2, ensure_ascii=False)}\n</behavioral_metadata>")
        return "\n".join(parts)

@dataclass
class Prompt:
    id: str
    text: str
    fingerprint: dict = field(default_factory=dict)

@dataclass
class Scenario:
    id: str
    category: str
    initial_prompt: str
    context_note: str = ""

class LLM:
    def __init__(self, model=None, max_concurrent=10, retries=3, log_calls=False):
        self.model = model or MODEL
        self.client = AsyncOpenAI(base_url=API_URL, api_key=API_KEY)
        self.sem = asyncio.Semaphore(max_concurrent)
        self.retries = retries
        self.calls = 0
        self.tokens = 0
        # Optional JSONL call logging (ported from utils/llm_client.py)
        self._log_file = None
        if log_calls:
            import time
            log_dir = OUT_DIR / "llm_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = log_dir / f"calls_{self.model}_{int(time.time())}.jsonl"

    def _log_call(self, record: dict):
        if not self._log_file:
            return
        import time
        record["timestamp"] = time.time()
        record["call_index"] = self.calls
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write LLM log: %s", e)

    async def chat(self, messages, temperature=0.7, max_tokens=4096,
                   json_mode=False, return_thinking=False,
                   call_type: str = "chat") -> str | tuple[str, str]:
        """Send a chat request. If return_thinking=True, returns (content, thinking) tuple."""
        import time as _time
        async with self.sem:
            for attempt in range(1, self.retries + 1):
                try:
                    kw = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                    if json_mode:
                        kw["response_format"] = {"type": "json_object"}
                    t0 = _time.time()
                    r = await self.client.chat.completions.create(**kw)
                    elapsed = _time.time() - t0
                    self.calls += 1
                    usage = {}
                    if r.usage:
                        self.tokens += r.usage.total_tokens
                        usage = {"prompt": r.usage.prompt_tokens,
                                 "completion": r.usage.completion_tokens,
                                 "total": r.usage.total_tokens}
                    msg = r.choices[0].message
                    content = msg.content or ""
                    thinking = ""
                    if return_thinking:
                        thinking = getattr(msg, "reasoning_content", None) or ""
                    self._log_call({
                        "call_type": call_type, "model": self.model,
                        "temperature": temperature, "max_tokens": max_tokens,
                        "output_preview": content[:300], "usage": usage,
                        "elapsed_s": round(elapsed, 2), "attempt": attempt,
                    })
                    if return_thinking:
                        return content, thinking
                    return content
                except Exception as e:
                    logger.warning("LLM attempt %d/%d: %s", attempt, self.retries, e)
                    if attempt < self.retries:
                        await asyncio.sleep(2 * attempt)
                    else:
                        raise

    async def chat_json(self, messages, **kw) -> dict:
        text = await self.chat(messages, json_mode=True, **kw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s, e = text.find("{"), text.rfind("}") + 1
            if s >= 0 and e > s:
                return json.loads(text[s:e])
            raise

    @property
    def stats(self):
        s = {"calls": self.calls, "tokens": self.tokens}
        if self._log_file:
            s["log_file"] = str(self._log_file)
        return s

def load_personas(d=None) -> list[Persona]:
    """Load persona YAML files. Supports both raw and refined profiles.

    Refined profiles may have: behavioral_metadata, refined_summary, selected_prompts.
    """
    d = d or DATA_DIR / "profiles" / "yaml"
    out = []
    for p in sorted(Path(d).glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            # Build metadata dict from refined fields
            metadata = {}
            if raw.get("behavioral_metadata"):
                metadata["behavioral_metadata"] = raw["behavioral_metadata"]
            if raw.get("refined_summary"):
                metadata["refined_summary"] = raw["refined_summary"]

            out.append(Persona(
                id=raw.get("persona_id", p.stem),
                attributes=raw.get("attributes") or {},
                summary=raw.get("summary", ""),
                fingerprint=raw.get("fingerprint") or {},
                metadata=metadata,
                selected_prompts=raw.get("selected_prompts") or [],
            ))
        except Exception as e:
            logger.warning("skip %s: %s", p, e)
    logger.info("loaded %d personas", len(out))
    return out

def load_prompts(path=None) -> list[Prompt]:
    path = path or DATA_DIR / "initial_prompts" / "prompts_mixed_taged.jsonl"
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                out.append(Prompt(id=d["prompt_id"], text=d["prompt_text"], fingerprint=d.get("fingerprint", {})))
            except json.JSONDecodeError:
                continue
    logger.info("loaded %d prompts", len(out))
    return out

def match_prompts(persona: Persona, prompts: list[Prompt], top_k=200) -> list[Prompt]:
    """Score prompts by fingerprint overlap, return top-k."""
    def _n(v): return v.strip().lower() if isinstance(v, str) else ""
    def _nl(v): return [_n(x) for x in v] if isinstance(v, list) else [_n(v)]

    scored = []
    pd, pr, pe = _nl(persona.fingerprint.get("domain", ["general"])), \
                 _nl(persona.fingerprint.get("region", ["GLOBAL"])), \
                 _n(persona.fingerprint.get("expertise_level", "mid"))
    pt = [_n(t) for t in persona.fingerprint.get("preferred_task_types", [])]

    for p in prompts:
        s = 0.0
        fd = _nl(p.fingerprint.get("domain", []))
        if set(pd) & set(fd): s += 3.0
        elif "general" in fd or "general" in pd: s += 1.5
        if _n(p.fingerprint.get("register", "")) == _n(persona.fingerprint.get("register", "")): s += 1.5
        fr = _nl(p.fingerprint.get("region", []))
        if set(pr) & set(fr) or "global" in fr or "global" in pr: s += 1.0
        fe = _n(p.fingerprint.get("expertise_level_implied", ""))
        if fe == pe: s += 2.0
        ft = _n(p.fingerprint.get("task_type", ""))
        if ft in pt: s += 2.5
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:top_k]]

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def fmt_conversation(conv: list[dict], up_to=None) -> str:
    """Format conversation as User:/Assistant: text for prompt injection."""
    parts = []
    for m in (conv[:up_to] if up_to else conv):
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts)