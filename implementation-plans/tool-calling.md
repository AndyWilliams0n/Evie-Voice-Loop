# Evie Voice Loop: Tool-Calling Feature

## Context

Evie uses Gemma 4 E4B (4B params) locally via mlx_vlm. The user wants the voice assistant to perform simple actions — read files, write JSON, execute build harness commands — with near-100% reliability. A 4B model is unreliable at native structured output (Gemma's `call:function{args}` format), so we use a **keyword pre-filter + LLM extraction** hybrid that keeps the model doing what it's good at (conversation, simple extraction) while Python handles all structured I/O.

## Approach: Keyword Pre-filter + LLM Extraction

```
transcribe → keyword match on transcript
  ├── no match → normal conversation (zero added latency)
  └── match → LLM extracts params → Python executes → LLM narrates result
```

**Why not native Gemma 4 tool calling?** The `<|tool_call>call:name{args}<tool_call|>` format exists in mlx_lm's parsers, but 4B models produce it inconsistently. Classification + extraction is more reliable and degrades gracefully (falls through to conversation on any failure).

## Files to modify

| File | Change |
|------|--------|
| `evie-mac.py` | Add `--tools` flag, `load_tools()`, `match_intent()`, `extract_params()`, `execute_tool()`, `format_tool_result()`. Modify `process_utterance()` to insert tool routing. |
| `tools.json` (new) | Tool registry: triggers, param extraction prompts, handlers, aliases, allowed commands |
| `SOUL.md` | Add one line about tool capabilities so LLM knows it can act |

## Implementation Steps

### Step 1: Create `tools.json`

Tool registry with three builtin tools:

```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read contents of a file",
      "triggers": ["read file", "read my", "what's in", "show me", "open the", "check my", "look at"],
      "params": {
        "path": {
          "type": "string",
          "extraction_prompt": "What file does the user want to read? Output ONLY the file path or short description, nothing else."
        }
      },
      "handler": "builtin:read_file",
      "confirm": false
    },
    {
      "name": "write_json",
      "description": "Add an item to a JSON array file",
      "triggers": ["add to", "save to", "write to", "append to", "put this in", "add item", "remember this"],
      "params": {
        "path": {
          "type": "string",
          "extraction_prompt": "What JSON file should this be saved to? Output ONLY the file path, nothing else."
        },
        "data": {
          "type": "string",
          "extraction_prompt": "What data should be saved? Output ONLY the data content, nothing else."
        }
      },
      "handler": "builtin:write_json",
      "confirm": true
    },
    {
      "name": "run_command",
      "description": "Execute a predefined command",
      "triggers": ["run the", "execute", "start the", "do a build", "check the pipeline", "trigger"],
      "params": {
        "command": {
          "type": "string",
          "extraction_prompt": "What command does the user want to run? Output ONLY the command name, nothing else.",
          "allowed_values": ["build", "test", "deploy", "pipeline_state", "pipeline_next"]
        }
      },
      "handler": "builtin:run_command",
      "confirm": true
    }
  ],
  "aliases": {
    "shopping list": "~/notes/shopping.json",
    "my notes": "~/notes/notes.json",
    "todo": "~/notes/todo.json"
  },
  "allowed_commands": {
    "build": "cd ~/project && make build",
    "test": "cd ~/project && make test",
    "pipeline_state": "curl -s http://localhost:3000/api/pipeline/state",
    "pipeline_next": "curl -s -X POST http://localhost:3000/api/pipeline/next"
  }
}
```

- **Triggers** are 2+ word phrases to avoid false positives ("read my" not "read")
- **Aliases** map spoken shorthand to file paths (skips LLM extraction)
- **allowed_commands** is a strict whitelist — no arbitrary shell execution

### Step 2: Add `--tools` CLI flag

In argparse block (~line 137), add:
```python
p.add_argument('--tools', '--no-tools', action=argparse.BooleanOptionalAction, default=False)
```

### Step 3: Add `load_tools()` (~line 47, after `load_system_prompt`)

Load tools.json once at startup. Returns empty config if file missing.

### Step 4: Add `match_intent(heard, tools_config)`

- Lowercase transcript
- Check aliases first — resolve to file path if matched
- For each tool, check if any trigger phrase is a substring
- Longest matching trigger wins (most specific)
- Return `(tool_def, resolved_alias)` or `(None, None)`

### Step 5: Add `extract_params(tool_def, heard, resolved_alias)`

- For each param in tool definition, run a short LLM call using extraction_prompt
- **Skip LLM call** if alias already resolved the path param
- Use `temperature=0.1, max_tokens=60` for near-deterministic extraction
- For params with `allowed_values`, fuzzy-match output against the list
- Return `None` on extraction failure (triggers fallback to conversation)

### Step 6: Add `execute_tool(tool_def, params, tools_config)`

Three builtin handlers:
- **read_file**: Expand `~`, check exists, read contents (cap 2000 chars), return `(True, content)`
- **write_json**: Load existing array or create new, append data, write back, return `(True, "Saved")`
- **run_command**: Look up in allowed_commands whitelist, subprocess.run with 30s timeout, return `(True, stdout)`

All return `(success: bool, result: str)`. All errors caught and returned as `(False, error_message)`.

### Step 7: Add `format_tool_result(tool_name, params, success, result, heard)`

Single LLM call that narrates the tool result in Evie's voice. Includes `_sys_messages()` so persona is maintained. This **replaces** the normal conversation LLM call, so net latency is ~0ms extra.

### Step 8: Modify `process_utterance()`

Insert after transcription, before normal LLM call:

```python
if args.tools:
    tool_def, alias_path = match_intent(heard, tools_config)
    if tool_def is not None:
        params = extract_params(tool_def, heard, alias_path)
        if params is not None:
            success, result = execute_tool(tool_def, params, tools_config)
            response = format_tool_result(tool_def['name'], params, success, result, heard)
            # speak, update history, return early
        # else: fall through to normal conversation
```

On any failure, **falls through** to normal conversation — safe default.

## Latency Impact

| Path | Extra LLM calls | Added latency |
|------|-----------------|---------------|
| Normal conversation (no match) | 0 | **Zero** (keyword check ~0.1ms) |
| Tool with alias-resolved path | 0 extraction + 1 narration | **~0ms** (narration replaces normal call) |
| Tool needing extraction | 1-2 extraction + 1 narration | **~200-400ms** extra |

## Error Handling

- tools.json missing → `--tools` is a no-op
- False positive keyword match → extraction fails → falls through to conversation
- LLM extraction garbage → allowed_values fuzzy match fails → falls through
- File not found → `(False, "File not found")` → LLM narrates error
- JSON parse error → `(False, "Invalid JSON")` → LLM narrates error
- Command not in whitelist → `(False, "Unknown command")` → LLM narrates error
- Command timeout (30s) → `(False, "Timed out")` → LLM narrates error

## Verification

1. Run `python evie-mac.py --tools --no-tts` for text-only testing
2. Test trigger phrases: "read my shopping list", "add eggs to my shopping list", "run the build"
3. Test non-triggers: "I read a great book", "what's the weather" — should NOT match
4. Test extraction failures: vague requests should fall through to conversation
5. Test file I/O: create a test JSON array, add items, read it back

## Future Extensions

- **Confirmation dialog**: For `confirm: true` tools, Evie asks "okay?" and waits for next utterance
- **MCP integration**: `allowed_commands` entries with `mcp:` prefix route to AssistAgent HTTP calls
- **Tool context in history**: Raw results stored so follow-up questions work
