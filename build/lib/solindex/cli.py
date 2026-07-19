#!/usr/bin/env python3
"""
Runtime State Indexer (solindex) v1.0

Single responsibility:

    Given a storage variable belonging to a specific contract, enumerate
    every function capable of writing that variable.

This tool performs indexing only. It performs no reasoning, no
interpretation, and no execution reconstruction. The analyst supplies the
reasoning; solindex exists solely to eliminate repetitive manual traversal
of Solidity source when answering one question: "which functions can
modify this storage variable?"

This intentionally REPLACES the earlier Runtime State Extractor, which
attempted to reconstruct full execution traces (reads, writes, control
flow, internal/external call graphs). That approach was unnecessarily
complex, difficult to verify, and difficult to trust. solindex answers a
single, narrow, mechanically-checkable question instead.

Usage:
    solindex <project_root> <contract> <variable> [--locations]

Examples:
    solindex contracts/ ActivePool.sol collBalance
    solindex contracts/ BorrowerOperations.sol troveManager
    solindex contracts/ ActivePool.sol collBalance --locations

<contract> may be:
    - a .sol filename (searched for under <project_root>), or
    - a bare contract/library name (searched for under <project_root>), or
    - an explicit path to a .sol file (relative to the cwd or the project
      root)

The contract is mandatory. State variable names are not globally unique
across a codebase, so solindex never guesses which contract owns the
variable you mean - a query with only a variable name and no contract is
rejected outright (see `--help` / the "ambiguous query" error below).

--------------------------------------------------------------------------
Writer definition
--------------------------------------------------------------------------
A writer is any function (including the constructor, fallback, and
receive) whose body - or whose body PLUS any modifier applied to it -
contains one of:

    - direct assignment            x = value;
    - compound assignment          x += value;  x -= value;  etc.
    - increment / decrement        x++;  --x;
    - mapping / nested-mapping updates through the above forms
    - struct member writes through the above forms
    - `delete x;`
    - `x.push(...)` / `x.pop()`
    - any of the above reached through a `storage` reference (an alias)
      that ultimately roots back to the queried variable

...where `x` (after resolving any chain of struct-field/array-index/
mapping-key access down to its base identifier, and resolving through any
storage-reference aliasing) names the queried storage variable.

--------------------------------------------------------------------------
Non-goals (deliberately out of scope)
--------------------------------------------------------------------------
solindex never: reconstructs execution, builds execution traces, builds
call graphs, builds semantic graphs, infers protocol behaviour, infers
invariants, infers guarantees, infers security properties, infers
exploitability, ranks functions, or performs vulnerability analysis. If a
write cannot be resolved with certainty, it is not reported - deterministic
correctness matters more than completeness. See "Known limitations" in the
README for the specific, documented cases this conservatively omits.
"""

import argparse
import bisect
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


# ==========================================================================
# Layout-preserving comment / string-literal masking
# ==========================================================================
# Both passes replace characters with spaces (never delete them) so that
# every remaining character keeps its ORIGINAL absolute offset and line
# number in the file. This is what makes `--locations` possible without a
# separate line-number reconstruction pass.

def strip_comments_preserve_layout(src: str) -> str:
    out = list(src)

    for m in re.finditer(r'/\*.*?\*/', src, flags=re.DOTALL):
        for i in range(m.start(), m.end()):
            if out[i] != '\n':
                out[i] = ' '
    for m in re.finditer(r'//[^\n]*', src):
        for i in range(m.start(), m.end()):
            out[i] = ' '
    return ''.join(out)


def mask_string_literals(text: str) -> str:
    """Blank out the contents of "..." / '...' string literals so that
    identifier-looking substrings inside string/error-message literals are
    never mistaken for a write target. Length- and line-preserving."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in ('"', "'"):
            quote = c
            j = i + 1
            while j < n and text[j] != quote:
                if text[j] == '\\':
                    j += 2
                    continue
                j += 1
            for k in range(i + 1, min(j, n)):
                if out[k] != '\n':
                    out[k] = ' '
            i = j + 1
        else:
            i += 1
    return ''.join(out)


# ==========================================================================
# Balanced-brace / paren helpers
# ==========================================================================

def find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced braces in source")


def find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced parens in source")


def find_stmt_end(text: str, start: int) -> int:
    """Index of the top-level ';' terminating the statement starting at
    `start` (skipping over nested (), [], {})."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ';' and depth == 0:
            return i
        i += 1
    return n


def top_level_split_op(text: str, ops) -> Optional[Tuple[int, str]]:
    """First top-level occurrence of any op in `ops`, ignoring occurrences
    inside (), [], {}, and avoiding false positives on ==, !=, <=, >=, =>."""
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in '([{':
            depth += 1
            i += 1
            continue
        if c in ')]}':
            depth -= 1
            i += 1
            continue
        if depth == 0:
            for op in ops:
                if text[i:i + len(op)] == op:
                    if op == '=':
                        prev = text[i - 1] if i > 0 else ''
                        nxt = text[i + 1] if i + 1 < n else ''
                        if prev in '=!<>' or nxt in '=>':
                            continue
                    return (i, op)
        i += 1
    return None


def skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in ' \t\r\n':
        i += 1
    return i


ASSIGN_OPS = ['<<=', '>>=', '+=', '-=', '*=', '/=', '%=', '|=', '&=', '^=', '=']

# Matches local storage-reference declarations, e.g.
# `Position storage p = positions[id]`. Known limitation: the type token
# must be a single bare identifier, so declarations like
# `mapping(address => uint) storage m2 = m1;` are not recognized (see
# README "Known limitations").
STORAGE_ALIAS_RE = re.compile(r'^(\w+)\s+storage\s+(\w+)\s*=\s*(.+)$', re.DOTALL)


def base_identifier(text: str) -> Optional[str]:
    """The leading identifier of a write-target/expression fragment, e.g.
    "positions[user][id]" -> "positions", "p.debt" -> "p". A write target
    always begins with an identifier in Solidity (no other expression form
    is a valid assignment/delete/push/pop target), so this is sufficient
    to find a chain's root without walking its index/field suffixes."""
    m = re.match(r'\s*([A-Za-z_]\w*)', text)
    return m.group(1) if m else None


# ==========================================================================
# Declaration parsing (state variables)
# ==========================================================================

def strip_default_value(stmt: str) -> str:
    """Strip a trailing `= <default expr>` from a declaration statement,
    without being fooled by ==, <=, >=, !=, or the '=>' inside mapping
    types."""
    m = re.search(r'(?<![=!<>])=(?![=>])', stmt)
    if m:
        return stmt[:m.start()].strip()
    return stmt.strip()


def parse_decl_name(stmt: str) -> Optional[Tuple[str, str, List[str]]]:
    """Parse a single declaration statement (no trailing ';') and return
    (name, type, modifiers), or None if it cannot be parsed as a
    `<type> <modifiers...> <name>` declaration."""
    stmt = strip_default_value(stmt)
    stmt = stmt.strip()
    if not stmt:
        return None

    if stmt.startswith('mapping'):
        paren_idx = stmt.find('(')
        if paren_idx == -1:
            return None
        close_idx = find_matching_paren(stmt, paren_idx)
        type_token = stmt[:close_idx + 1]
        rest = stmt[close_idx + 1:].strip()
        rest_tokens = rest.split()
        if not rest_tokens:
            return None
        name = rest_tokens[-1]
        modifiers = rest_tokens[:-1]
        return (name, type_token, modifiers)

    tokens = stmt.split()
    if len(tokens) < 2:
        return None

    if tokens[0] == 'address' and len(tokens) >= 3 and tokens[1] == 'payable':
        type_token = 'address payable'
        rest_tokens = tokens[2:]
    else:
        type_token = tokens[0]
        rest_tokens = tokens[1:]

    if not rest_tokens:
        return None

    name = rest_tokens[-1]
    if not re.match(r'^\w+$', name):
        return None
    modifiers = rest_tokens[:-1]
    return (name, type_token, modifiers)


def strip_blocks(text: str, keyword_pattern: str) -> str:
    """Remove every `<keyword> ... { ... }` (or `<keyword> ...;` body-less)
    span from text, for keywords matching keyword_pattern. Used to blank
    out function/modifier/struct/enum bodies before scanning a contract
    body for top-level state variable declarations."""
    while True:
        m = re.search(keyword_pattern, text)
        if not m:
            break
        brace_idx = text.find('{', m.start())
        semi_idx = text.find(';', m.start())
        if brace_idx == -1 or (semi_idx != -1 and semi_idx < brace_idx):
            if semi_idx == -1:
                text = text[:m.start()] + text[m.end():]
                continue
            text = text[:m.start()] + text[semi_idx + 1:]
            continue
        close_idx = find_matching_brace(text, brace_idx)
        text = text[:m.start()] + text[close_idx + 1:]
    return text


STATEMENT_SKIP_PREFIXES = (
    'function', 'event', 'error', 'using', 'import', 'pragma',
    'modifier', 'constructor', 'fallback', 'receive',
)


def extract_state_var_names(body: str, include_constants: bool = False) -> List[Tuple[str, str]]:
    """Given a contract body, strip function/modifier/struct/enum blocks,
    then parse remaining top-level statements for state variable
    declarations. Returns ordered (name, type) pairs, skipping constants
    and immutables by default (never storage-backed, so never "writable"
    in the writer-indexing sense this tool cares about) - pass
    `include_constants=True` to keep them too (needed by the trace layer:
    a `.call()` receiver's declared type matters for resolution
    regardless of mutability, e.g. `ITroveNFT public immutable troveNFT`)."""
    text = body
    text = strip_blocks(text, r'\b(function|constructor|modifier|fallback|receive)\b')
    text = strip_blocks(text, r'\b(struct|enum)\b')

    names = []
    for raw_stmt in text.split(';'):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        if any(stmt.startswith(kw) for kw in STATEMENT_SKIP_PREFIXES):
            continue
        parsed = parse_decl_name(stmt)
        if parsed is None:
            continue
        name, type_token, modifiers = parsed
        if not include_constants and ('constant' in modifiers or 'immutable' in modifiers):
            continue
        names.append((name, type_token))
    return names


# ==========================================================================
# Contract extraction + same-file inheritance
# ==========================================================================

CONTRACT_HEADER_RE = re.compile(
    r'\b(abstract\s+contract|contract|interface|library)\s+(\w+)'
    r'(\s+is\s+([^{]+))?\s*\{'
)


def extract_contracts(src: str):
    """Returns ordered list of dicts:
    {kind, name, bases, body, body_start, order}
    `body_start` is the ABSOLUTE offset (into `src`) where `body` begins -
    this is what lets every downstream offset stay traceable back to a
    real file:line."""
    contracts = []
    order = 0
    for m in CONTRACT_HEADER_RE.finditer(src):
        kind = m.group(1)
        name = m.group(2)
        bases_str = m.group(4) or ''
        bases = re.findall(r'([A-Za-z_]\w*)\s*(?:\([^)]*\))?', bases_str)
        bases = [b for b in bases if b]
        open_brace_idx = m.end() - 1
        close_idx = find_matching_brace(src, open_brace_idx)
        body = src[open_brace_idx + 1:close_idx]
        contracts.append({
            'kind': kind,
            'name': name,
            'bases': bases,
            'body': body,
            'body_start': open_brace_idx + 1,
            'order': order,
        })
        order += 1
    return contracts


def collect_state_var_names(contract, contracts_by_name, seen=None) -> List[Tuple[str, str]]:
    """Recursively resolve in-file base contracts (declaration order),
    then this contract's own state variables. Bases not defined in this
    file are out of scope (single-file tool)."""
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return []
    seen.add(contract['name'])

    names: List[Tuple[str, str]] = []
    for base_name in contract['bases']:
        base = contracts_by_name.get(base_name)
        if base is None:
            continue
        names.extend(collect_state_var_names(base, contracts_by_name, seen))
    names.extend(extract_state_var_names(contract['body']))
    return names


# ==========================================================================
# Function / modifier extraction
# ==========================================================================

FUNC_HEADER_RE = re.compile(r'\bfunction\s+(\w+)\s*\(')
CTOR_HEADER_RE = re.compile(r'\bconstructor\s*\(')
SPECIAL_HEADER_RE = re.compile(r'\b(fallback|receive)\s*\(')
MODIFIER_HEADER_RE = re.compile(r'\bmodifier\s+(\w+)\s*\(([^)]*)\)\s*\{')

SKIP_HDR_WORDS = {
    'public', 'external', 'internal', 'private', 'view', 'pure',
    'payable', 'virtual', 'override', 'returns',
}


def extract_functions_from_body(body: str, body_start_abs: int):
    """Parse `function NAME(...) { ... }` / `constructor(...) { ... }` /
    `fallback(...) { ... }` / `receive(...) { ... }` declarations out of a
    contract body. Returns a list of dicts (declaration order preserved):
    {name, modifiers, body, body_start_abs, order}. `body` is None for a
    body-less (interface/abstract) signature. `body_start_abs` is the
    ABSOLUTE offset of `body`'s first character in the original source
    file, for `--locations`."""
    headers = []
    for m in FUNC_HEADER_RE.finditer(body):
        headers.append((m.start(), m.group(1)))
    for m in CTOR_HEADER_RE.finditer(body):
        headers.append((m.start(), 'constructor'))
    for m in SPECIAL_HEADER_RE.finditer(body):
        headers.append((m.start(), m.group(1)))
    headers.sort(key=lambda h: h[0])

    funcs = []
    order = 0
    for start, name in headers:
        header_start_abs = body_start_abs + start
        paren_open = body.find('(', start)
        paren_close = find_matching_paren(body, paren_open)
        rest_start = paren_close + 1
        brace_idx = body.find('{', rest_start)
        semi_idx = body.find(';', rest_start)
        if brace_idx == -1 or (semi_idx != -1 and semi_idx < brace_idx):
            header_tail = body[rest_start:semi_idx] if semi_idx != -1 else body[rest_start:]
            func_body = None
            func_body_start_abs = None
            end_abs = body_start_abs + (semi_idx + 1 if semi_idx != -1 else len(body))
        else:
            header_tail = body[rest_start:brace_idx]
            close_b = find_matching_brace(body, brace_idx)
            func_body = body[brace_idx + 1:close_b]
            func_body_start_abs = body_start_abs + brace_idx + 1
            end_abs = body_start_abs + close_b + 1

        htail = re.sub(r'\breturns\s*\([^)]*\)', '', header_tail)
        vis_m = re.search(r'\b(public|external|internal|private)\b', htail)
        visibility = vis_m.group(1) if vis_m else 'public'
        modifiers = []
        for mm in re.finditer(r'\b(\w+)\s*(\(([^)]*)\))?', htail):
            wname = mm.group(1)
            if not wname or wname in SKIP_HDR_WORDS:
                continue
            modifiers.append((wname, mm.group(3) or ''))

        funcs.append({
            'name': name, 'modifiers': modifiers, 'visibility': visibility,
            'body': func_body, 'body_start_abs': func_body_start_abs,
            'header_start_abs': header_start_abs, 'end_abs': end_abs,
            'order': order,
        })
        order += 1
    return funcs


def extract_modifiers_from_body(body: str, body_start_abs: int) -> Dict[str, dict]:
    """Returns dict: modifier_name -> {body, body_start_abs, header_start_abs,
    end_abs}."""
    mods = {}
    for m in MODIFIER_HEADER_RE.finditer(body):
        name = m.group(1)
        brace_idx = m.end() - 1
        close = find_matching_brace(body, brace_idx)
        mods[name] = {
            'body': body[brace_idx + 1:close],
            'body_start_abs': body_start_abs + brace_idx + 1,
            'header_start_abs': body_start_abs + m.start(),
            'end_abs': body_start_abs + close + 1,
        }
    return mods


def collect_functions_and_modifiers(contract, contracts_by_name, seen=None):
    """Recursively resolve in-file base contracts (declaration order),
    then this contract's own functions/modifiers. Own-contract functions
    shadow base functions of the same name (override)."""
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return [], {}
    seen.add(contract['name'])

    funcs_by_name: Dict[str, dict] = {}
    mods_by_name: Dict[str, dict] = {}
    order_list = []

    for base_name in contract['bases']:
        base = contracts_by_name.get(base_name)
        if base is None:
            continue
        base_order, base_mods = collect_functions_and_modifiers(base, contracts_by_name, seen)
        mods_by_name.update(base_mods)
        for f in base_order:
            funcs_by_name[f['name']] = f
        order_list.extend(base_order)

    own_funcs = extract_functions_from_body(contract['body'], contract['body_start'])
    own_mods = extract_modifiers_from_body(contract['body'], contract['body_start'])
    mods_by_name.update(own_mods)
    for f in own_funcs:
        if f['name'] in funcs_by_name:
            order_list = [x for x in order_list if x['name'] != f['name']]
        funcs_by_name[f['name']] = f
        order_list.append(f)

    return order_list, mods_by_name


def collect_all_functions_multi(contract, contracts_by_name, seen=None) -> List[dict]:
    """Used ONLY by the --extract function-extraction feature (never by the
    writer-indexing path above). Recursively walks same-file base contracts
    exactly like collect_functions_and_modifiers, but deliberately does NOT
    dedupe by name: every declaration of a given function name is kept,
    including Solidity overloads and interface/abstract bodyless signatures
    that a concrete override also happens to satisfy. The caller is
    responsible for preferring bodied implementations over bodyless ones."""
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return []
    seen.add(contract['name'])

    funcs: List[dict] = []
    for base_name in contract['bases']:
        base = contracts_by_name.get(base_name)
        if base is None:
            continue
        funcs.extend(collect_all_functions_multi(base, contracts_by_name, seen))
    funcs.extend(extract_functions_from_body(contract['body'], contract['body_start']))
    return funcs


# ==========================================================================
# Writer detection
# ==========================================================================
# The whole engine below answers exactly one question per function: does
# ANY statement in this function's body (or in a modifier applied to it)
# write the queried variable? No trace tree, no read tracking, no call
# classification is built - control structures (if/for/while/do/
# unchecked/bare blocks) are only ever walked to reach every statement;
# their conditions are never inspected, because a Solidity condition
# cannot itself perform a storage write.
#
# Function and modifier bodies are scanned INDEPENDENTLY: a storage alias
# declared inside a modifier is not visible to the function it guards, and
# vice versa. This is a documented simplification, not an oversight - see
# README "Known limitations".

class _FoundWrite(Exception):
    def __init__(self, offset: int):
        self.offset = offset


def _resolve_root(name: str, aliases: Dict[str, str]) -> str:
    seen = set()
    root = name
    while root in aliases and root not in seen:
        seen.add(root)
        root = aliases[root]
    return root


PUSH_POP_RE = re.compile(r'^[A-Za-z_]\w*[\w.\[\]\s]*\.\s*(push|pop)\s*\(')


def scan_for_write_offset(body: str, body_start_abs: int, target_var: str) -> Optional[int]:
    """Returns the absolute source offset of the first statement in `body`
    that writes `target_var`, or None if there is no such statement."""
    aliases: Dict[str, str] = {}

    def check(chain_text: str, abs_off: int) -> None:
        ident = base_identifier(chain_text)
        if ident is None:
            return
        if _resolve_root(ident, aliases) == target_var:
            raise _FoundWrite(abs_off)

    def process_stmt(stmt_text: str, stmt_abs_start: int) -> None:
        text = stmt_text.strip()
        if not text:
            return
        lead = len(stmt_text) - len(stmt_text.lstrip())
        base_off = stmt_abs_start + lead

        alias_m = STORAGE_ALIAS_RE.match(text)
        if alias_m:
            alias_name, expr_text = alias_m.group(2), alias_m.group(3)
            ident = base_identifier(expr_text)
            if ident is not None:
                aliases[alias_name] = _resolve_root(ident, aliases)
            return

        delete_m = re.match(r'delete\s+(.+)$', text, re.DOTALL)
        if delete_m:
            check(delete_m.group(1), base_off + delete_m.start(1))
            return

        split = top_level_split_op(text, ASSIGN_OPS)
        if split:
            pos, _op = split
            lhs = text[:pos].strip()
            if lhs.startswith('(') and lhs.endswith(')'):
                inner = lhs[1:-1]
                depth = 0
                cur_start = 0
                for idx, c in enumerate(inner):
                    if c in '([{':
                        depth += 1
                    elif c in ')]}':
                        depth -= 1
                    if c == ',' and depth == 0:
                        part = inner[cur_start:idx]
                        part_stripped = part.strip()
                        if part_stripped:
                            extra = len(part) - len(part.lstrip())
                            check(part_stripped, base_off + 1 + cur_start + extra)
                        cur_start = idx + 1
                part = inner[cur_start:]
                part_stripped = part.strip()
                if part_stripped:
                    extra = len(part) - len(part.lstrip())
                    check(part_stripped, base_off + 1 + cur_start + extra)
            else:
                check(lhs, base_off)
            return

        prefix_m = re.match(r'^(\+\+|--)(.+)$', text)
        if prefix_m:
            check(prefix_m.group(2), base_off + prefix_m.start(2))
            return
        postfix_m = re.match(r'^(.+?)(\+\+|--)$', text)
        if postfix_m:
            check(postfix_m.group(1), base_off)
            return

        if PUSH_POP_RE.match(text):
            check(text, base_off)
            return

    def walk_block(text: str, abs_start: int) -> None:
        i = 0
        n = len(text)
        while i < n:
            i = skip_ws(text, i)
            if i >= n:
                break
            if text[i] == ';':
                i += 1
                continue
            if text[i] == '{':
                close = find_matching_brace(text, i)
                walk_block(text[i + 1:close], abs_start + i + 1)
                i = close + 1
                continue
            if re.match(r'(if|for|while)\b', text[i:]):
                i = handle_control(text, i, abs_start)
                continue
            if re.match(r'unchecked\b', text[i:]):
                j = skip_ws(text, i + len('unchecked'))
                if j < n and text[j] == '{':
                    close = find_matching_brace(text, j)
                    walk_block(text[j + 1:close], abs_start + j + 1)
                    i = close + 1
                    continue
            if re.match(r'do\b', text[i:]):
                j = skip_ws(text, i + 2)
                if j < n and text[j] == '{':
                    close = find_matching_brace(text, j)
                    walk_block(text[j + 1:close], abs_start + j + 1)
                    k = skip_ws(text, close + 1)
                    if re.match(r'while\b', text[k:]):
                        wp = skip_ws(text, k + 5)
                        wclose = find_matching_paren(text, wp)
                        stmt_end = find_stmt_end(text, wclose + 1)
                        i = stmt_end + 1
                        continue
                    i = close + 1
                    continue
            stmt_end = find_stmt_end(text, i)
            process_stmt(text[i:stmt_end], abs_start + i)
            i = stmt_end + 1

    def handle_control(text: str, i: int, abs_start: int) -> int:
        kw = re.match(r'(if|for|while)\b', text[i:]).group(1)
        j = skip_ws(text, i + len(kw))
        paren_open = j
        close_p = find_matching_paren(text, paren_open)
        cond = text[paren_open + 1:close_p]
        k = skip_ws(text, close_p + 1)

        if kw == 'for':
            # init;cond;incr - init and incr are real statements (and, in
            # principle, could write storage); cond cannot.
            depth = 0
            start_idx = 0
            parts = []
            for idx, c in enumerate(cond):
                if c in '([{':
                    depth += 1
                elif c in ')]}':
                    depth -= 1
                if c == ';' and depth == 0:
                    parts.append((start_idx, cond[start_idx:idx]))
                    start_idx = idx + 1
            parts.append((start_idx, cond[start_idx:]))
            if len(parts) >= 1 and parts[0][1].strip():
                process_stmt(parts[0][1], abs_start + paren_open + 1 + parts[0][0])
            if len(parts) >= 3 and parts[2][1].strip():
                process_stmt(parts[2][1], abs_start + paren_open + 1 + parts[2][0])

        if k < len(text) and text[k] == '{':
            close_b = find_matching_brace(text, k)
            walk_block(text[k + 1:close_b], abs_start + k + 1)
            after = close_b + 1
        else:
            stmt_end = find_stmt_end(text, k)
            process_stmt(text[k:stmt_end], abs_start + k)
            after = stmt_end + 1

        if kw == 'if':
            p = skip_ws(text, after)
            if text[p:p + 4] == 'else' and (p + 4 >= len(text) or not (text[p + 4].isalnum() or text[p + 4] == '_')):
                p2 = skip_ws(text, p + 4)
                if re.match(r'if\b', text[p2:]):
                    after = handle_control(text, p2, abs_start)
                elif p2 < len(text) and text[p2] == '{':
                    close_b2 = find_matching_brace(text, p2)
                    walk_block(text[p2 + 1:close_b2], abs_start + p2 + 1)
                    after = close_b2 + 1
                else:
                    stmt_end2 = find_stmt_end(text, p2)
                    process_stmt(text[p2:stmt_end2], abs_start + p2)
                    after = stmt_end2 + 1
        return after

    try:
        walk_block(mask_string_literals(body), body_start_abs)
    except _FoundWrite as fw:
        return fw.offset
    return None


def scan_function_for_write(func: dict, mods_by_name: Dict[str, dict], target_var: str
                             ) -> Optional[Tuple[int, Optional[str]]]:
    """Checks the function's own body first, then each modifier applied to
    it (in listed order). Returns (absolute_offset, matched_modifier_name)
    where matched_modifier_name is None if the write was found directly in
    the function's own body, or None if no write was found at all."""
    if func['body'] is not None:
        hit = scan_for_write_offset(func['body'], func['body_start_abs'], target_var)
        if hit is not None:
            return (hit, None)
    for mod_name, _args in func['modifiers']:
        mod = mods_by_name.get(mod_name)
        if mod is None:
            continue
        hit = scan_for_write_offset(mod['body'], mod['body_start_abs'], target_var)
        if hit is not None:
            return (hit, mod_name)
    return None


# ==========================================================================
# Source-location resolution
# ==========================================================================

def compute_newline_offsets(src: str) -> List[int]:
    return [i for i, c in enumerate(src) if c == '\n']


def line_number(newline_offsets: List[int], abs_pos: int) -> int:
    return bisect.bisect_right(newline_offsets, abs_pos) + 1


# ==========================================================================
# Contract/file resolution
# ==========================================================================

def find_sol_files(root: str) -> List[str]:
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.sol'):
                matches.append(os.path.join(dirpath, fn))
    return sorted(matches)


def _load_and_locate(path: str, contract_arg: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
    except OSError as e:
        raise SystemExit(f"error: could not read {path}: {e}")

    src = strip_comments_preserve_layout(raw)
    contracts = extract_contracts(src)
    if not contracts:
        raise SystemExit(f"error: no contract/interface/library declarations found in {path}")
    contracts_by_name = {c['name']: c for c in contracts}

    if contract_arg.endswith('.sol'):
        stem = os.path.splitext(os.path.basename(contract_arg))[0]
        if stem in contracts_by_name and contracts_by_name[stem]['kind'] in ('contract', 'abstract contract'):
            target_name = stem
        else:
            concrete = [c for c in contracts if c['kind'] == 'contract']
            if len(concrete) == 1:
                target_name = concrete[0]['name']
            else:
                declared = ', '.join(c['name'] for c in contracts)
                raise SystemExit(
                    f"error: could not determine which contract in {path} to analyze "
                    f"(no contract named '{stem}' found; declarations in file: {declared}). "
                    f"Pass the contract name explicitly instead of the filename."
                )
    else:
        if contract_arg not in contracts_by_name:
            raise SystemExit(f"error: contract '{contract_arg}' not found in {path}")
        if contracts_by_name[contract_arg]['kind'] not in ('contract', 'abstract contract', 'library'):
            raise SystemExit(
                f"error: '{contract_arg}' in {path} is an interface, which cannot declare "
                f"storage variables - nothing to index"
            )
        target_name = contract_arg

    return path, raw, src, contracts_by_name, target_name


def resolve_target(project_root: str, contract_arg: str):
    """Returns (path, raw, src, contracts_by_name, target_contract_name).
    `raw` is the untouched file text (comments intact) and `src` is the
    comment/string-masked text used for scanning; both share identical
    character offsets, so any offset computed against `src` can be used to
    slice the corresponding, human-readable text out of `raw`."""
    direct_candidates = [contract_arg, os.path.join(project_root, contract_arg)]
    for c in direct_candidates:
        if os.path.isfile(c):
            return _load_and_locate(c, contract_arg)

    sol_files = find_sol_files(project_root)
    if not sol_files:
        raise SystemExit(f"error: no .sol files found under {project_root}")

    if contract_arg.endswith('.sol'):
        matches = [p for p in sol_files if os.path.basename(p) == contract_arg]
        if not matches:
            raise SystemExit(f"error: no file named '{contract_arg}' found under {project_root}")
        if len(matches) > 1:
            listing = '\n'.join(f"  - {p}" for p in matches)
            raise SystemExit(
                f"error: multiple files named '{contract_arg}' found under {project_root}; "
                f"pass a full relative path instead:\n{listing}"
            )
        return _load_and_locate(matches[0], contract_arg)

    name_re = re.compile(r'\b(?:abstract\s+contract|contract|library)\s+' + re.escape(contract_arg) + r'\b')
    matches = []
    for p in sol_files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                text = f.read()
        except OSError:
            continue
        if name_re.search(text):
            matches.append(p)
    if not matches:
        raise SystemExit(f"error: no contract/library named '{contract_arg}' found under {project_root}")
    if len(matches) > 1:
        listing = '\n'.join(f"  - {p}" for p in matches)
        raise SystemExit(
            f"error: contract '{contract_arg}' is declared in multiple files under {project_root}; "
            f"pass the file directly instead:\n{listing}"
        )
    return _load_and_locate(matches[0], contract_arg)


# ==========================================================================
# Function extraction (--extract) - a separate, independent feature
# ==========================================================================
# This is deliberately a self-contained "grep for one function" utility
# bolted onto the same CLI, not a part of writer-indexing. It shares only
# read-only parsing helpers with the code above (find_sol_files,
# extract_contracts, strip_comments_preserve_layout,
# collect_all_functions_multi) and is never invoked unless --extract is
# passed; the writer-indexing path above is completely unaffected by this
# section's presence.
#
# Known behavior this section is deliberately careful about: a codebase
# will often declare the same function name twice - once as a bodyless
# interface/abstract "virtual" signature, and once as the concrete
# override with the real implementation. A naive first-match grep would
# often surface the interface version instead of the real code. This
# extractor always prefers bodied (implemented) matches over bodyless
# ones, and if ONLY a bodyless signature can be found, it says so
# explicitly in the output rather than silently handing back an empty
# stub.

EXTRACT_SPEC_RE = re.compile(r'^(\w+)\.(\w+)$')


def _render_extract_block(project_root: str, sol_files: List[str], raw_spec: str) -> str:
    m = EXTRACT_SPEC_RE.match(raw_spec.strip())
    if not m:
        return (f"## `{raw_spec}`\n\n"
                f"_Malformed spec - expected `Contract.functionName`._")
    contract_name, func_name = m.group(1), m.group(2)

    # Same "which files declare this contract" search resolve_target()
    # uses for a bare contract name - deliberately excludes bare
    # `interface X` declarations, since we want the CONTRACT that
    # implements things, not the interface it satisfies.
    name_re = re.compile(r'\b(?:abstract\s+contract|contract|library)\s+' + re.escape(contract_name) + r'\b')
    matches = [p for p in sol_files if name_re.search(open(p, 'r', encoding='utf-8', errors='ignore').read())]

    if not matches:
        return (f"## `{raw_spec}`\n\n"
                f"_Not found: no contract, abstract contract, or library named `{contract_name}` "
                f"under `{project_root}`._")

    if len(matches) > 1:
        listing = "\n".join(f"- `{p}`" for p in matches)
        return (f"## `{raw_spec}`\n\n"
                f"_Ambiguous: `{contract_name}` is declared in multiple files:_\n\n{listing}\n\n"
                f"_Skipped - narrow `project_root` so only one file matches._")

    path = matches[0]
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    src = strip_comments_preserve_layout(raw)
    contracts = extract_contracts(src)
    contracts_by_name = {c['name']: c for c in contracts}
    contract = contracts_by_name.get(contract_name)
    if contract is None:
        return f"## `{raw_spec}`\n\n_Internal error: could not re-parse `{contract_name}` in `{path}`._"

    candidates = [f for f in collect_all_functions_multi(contract, contracts_by_name)
                  if f['name'] == func_name]

    if not candidates:
        return (f"## `{raw_spec}`\n\n"
                f"_Not found: no function named `{func_name}` in `{contract_name}` "
                f"(checked its own body and same-file base contracts) - file: `{path}`._")

    implemented = [f for f in candidates if f['body'] is not None]
    bodyless = [f for f in candidates if f['body'] is None]

    def render_one(f: dict) -> str:
        text = raw[f['header_start_abs']:f['end_abs']].strip('\n')
        return f"```solidity\n{text}\n```"

    parts = [f"## `{raw_spec}`", "", f"_Source: `{path}`_", ""]

    if implemented:
        if len(implemented) > 1:
            parts.append(f"> Found {len(implemented)} matching implementations for `{func_name}` "
                          f"(likely overloads) - showing all.\n")
            for idx, f in enumerate(implemented, 1):
                parts.append(f"**Match {idx}:**\n")
                parts.append(render_one(f))
        else:
            parts.append(render_one(implemented[0]))
    else:
        parts.append(
            f"> \u26a0\ufe0f No function **body** found for `{func_name}` in `{contract_name}` - only "
            f"an interface/abstract virtual declaration was located (shown below). This is very "
            f"likely NOT the real implementation. Check derived/overriding contracts elsewhere in "
            f"the codebase for the actual logic.\n"
        )
        for idx, f in enumerate(bodyless, 1):
            if len(bodyless) > 1:
                parts.append(f"**Declaration {idx}:**\n")
            parts.append(render_one(f))

    return "\n".join(parts)


def run_extract_mode(project_root: str, specs: List[str], output_path: Optional[str]) -> None:
    sol_files = find_sol_files(project_root)
    if not sol_files:
        raise SystemExit(f"error: no .sol files found under {project_root}")

    blocks = [_render_extract_block(project_root, sol_files, spec) for spec in specs]
    content = "# Extracted Functions\n\n" + "\n\n---\n\n".join(blocks) + "\n"

    path = output_path or "extracted_functions.md"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        raise SystemExit(f"error: could not write to {path}: {e}")
    print(f"Wrote {len(specs)} extracted function(s) to {path}")


# ==========================================================================
# ProjectIndex (spec Section 1)
# ==========================================================================

class ContractAmbiguityError(Exception):
    def __init__(self, name: str, records: List[dict]):
        self.name = name
        self.records = records
        listing = '\n'.join(f"  - {r['file_path']}" for r in records)
        super().__init__(
            f"error: contract '{name}' is declared in multiple files:\n{listing}"
        )


class ProjectIndex:
    """Built once per run. `by_name` maps a contract/interface/library name
    to its record, or to a list of records if the name is ambiguous
    (declared in more than one file) - see spec Section 1.3."""

    def __init__(self):
        self.by_name: Dict[str, object] = {}

    @classmethod
    def build(cls, project_root: str) -> 'ProjectIndex':
        idx = cls()
        groups: Dict[str, List[dict]] = {}
        for path in find_sol_files(project_root):
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read()
            src = strip_comments_preserve_layout(raw)
            for c in extract_contracts(src):
                c['file_path'] = path
                c['raw'] = raw
                groups.setdefault(c['name'], []).append(c)
        for name, records in groups.items():
            idx.by_name[name] = records[0] if len(records) == 1 else records
        return idx

    def resolve(self, name: str) -> Optional[dict]:
        entry = self.by_name.get(name)
        if entry is None:
            return None
        if isinstance(entry, list):
            raise ContractAmbiguityError(name, entry)
        return entry


# ==========================================================================
# Call target resolution (spec Section 4)
# ==========================================================================

LOW_LEVEL_CALLS = {'call', 'delegatecall', 'staticcall'}


def _gather_call_candidates(contract, func_name: str, project_index: ProjectIndex,
                             seen=None) -> List[Tuple[dict, dict]]:
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return []
    seen.add(contract['name'])

    out: List[Tuple[dict, dict]] = []
    for base_name in contract['bases']:
        base = project_index.resolve(base_name)
        if base is not None:
            out.extend(_gather_call_candidates(base, func_name, project_index, seen))
    for f in extract_functions_from_body(contract['body'], contract['body_start']):
        if f['name'] == func_name:
            out.append((contract, f))
    return out


def collect_call_candidates(contract, func_name: str, project_index: ProjectIndex
                             ) -> List[Tuple[dict, dict]]:
    """All same-name candidates on `contract` (own + same-file/cross-file
    inherited bases via `project_index`), preferring implemented (bodied)
    matches over bodyless ones - matches `_render_extract_block`'s
    implemented/bodyless policy (spec Section 4.4)."""
    out = _gather_call_candidates(contract, func_name, project_index)
    implemented = [(c, f) for c, f in out if f['body'] is not None]
    return implemented if implemented else out


def classify_kind(contract_kind: str, visibility: str) -> str:
    if contract_kind == 'library':
        return 'L'
    return 'X' if visibility in ('external', 'public') else 'I'


def make_call_node(kind: str, targets: List[str]) -> dict:
    # Frozen investigation-package schema (Section: functions/trace nodes)
    # only has a single "target" string per I/X/L node - when a call name
    # is overloaded, the first (implemented-preferred) match is used and
    # the other candidates are not represented.
    return {'type': kind, 'target': targets[0]}


BUILTIN_NAMESPACES = {'abi', 'block', 'msg', 'tx', 'type', 'super'}

STRUCT_DEF_RE = re.compile(r'\bstruct\s+(\w+)\s*\{([^}]*)\}')


def _extract_struct_defs(body: str) -> Dict[str, Dict[str, str]]:
    """Contract-local struct definitions: struct name -> {field: type}."""
    text = strip_blocks(body, r'\b(function|constructor|modifier|fallback|receive)\b')
    out: Dict[str, Dict[str, str]] = {}
    for m in STRUCT_DEF_RE.finditer(text):
        sname, inner = m.group(1), m.group(2)
        fields: Dict[str, str] = {}
        for raw in inner.split(';'):
            part = raw.strip()
            if not part:
                continue
            parsed = parse_decl_name(part)
            if parsed is None:
                continue
            fname, ftype, _mods = parsed
            fields[fname] = ftype
        out[sname] = fields
    return out


def _gather_struct_defs(contract, project_index: ProjectIndex, seen=None) -> Dict[str, Dict[str, str]]:
    """Struct definitions visible to `contract`: own + same-file/cross-file
    inherited bases, own contract's definitions taking precedence."""
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return {}
    seen.add(contract['name'])
    merged: Dict[str, Dict[str, str]] = {}
    for base_name in contract['bases']:
        base = project_index.resolve(base_name)
        if base is not None:
            merged.update(_gather_struct_defs(base, project_index, seen))
    merged.update(_extract_struct_defs(contract['body']))
    return merged


def _resolve_struct_fields(contract, type_name: str, project_index: ProjectIndex
                            ) -> Optional[Dict[str, str]]:
    """`type_name` may be a bare struct name (searched in `contract` + its
    bases) or a qualified `Contract.StructName` reference."""
    if '.' in type_name:
        cname, sname = type_name.split('.', 1)
        target = project_index.resolve(cname)
        if target is None:
            return None
        return _gather_struct_defs(target, project_index).get(sname)
    return _gather_struct_defs(contract, project_index).get(type_name)


def _local_type(name: str, shadow_stack: List[Dict[str, Optional[str]]]) -> Optional[str]:
    for scope in reversed(shadow_stack):
        if name in scope:
            return scope[name]
    return None


def find_declaring_contract(contract, var_name: str, project_index: ProjectIndex,
                             seen=None) -> Optional[dict]:
    """The contract that actually declares `var_name` (own body, not
    inherited) - searches bases first, then own body, matching
    `collect_state_var_names`'s resolution order."""
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return None
    seen.add(contract['name'])
    for base_name in contract['bases']:
        base = project_index.resolve(base_name)
        if base is not None:
            found = find_declaring_contract(base, var_name, project_index, seen)
            if found is not None:
                return found
    for name, _t in extract_state_var_names(contract['body'], include_constants=True):
        if name == var_name:
            return contract
    return None


def _owner_of(var_name: str, ctx: dict) -> Optional[str]:
    """Cached (within one trace_function call) lookup of the declaring
    contract's name for a W/R target - lets downstream consumers (e.g.
    the closure/investigation-package builder) correlate a node straight
    back to a specific `Contract.variable`, without re-walking
    inheritance themselves."""
    cache = ctx['owner_cache']
    if var_name in cache:
        return cache[var_name]
    found = find_declaring_contract(ctx['contract'], var_name, ctx['project_index'])
    owner = found['name'] if found is not None else None
    cache[var_name] = owner
    return owner


def _qualified_target(var_name: str, ctx: dict) -> str:
    """Frozen investigation-package schema: R/W nodes carry a single
    "Owner.varName" target string (not a separate target+owner pair).
    Falls back to the bare name in the (practically unreachable) case the
    declaring contract can't be found."""
    owner = _owner_of(var_name, ctx)
    return f"{owner}.{var_name}" if owner is not None else var_name


def _resolve_declared_type_call(declared_type: str, func_name: str,
                                 project_index: ProjectIndex) -> dict:
    """Spec Section 4.3 steps 2-4, given a declared type string."""
    target_contract = project_index.resolve(declared_type)
    if target_contract is None or target_contract['kind'] not in ('contract', 'abstract contract', 'library'):
        stripped = declared_type[1:] if re.match(r'^I[A-Z]', declared_type) else None
        resolved = project_index.resolve(stripped) if stripped else None
        if resolved is not None and resolved['kind'] in ('contract', 'abstract contract'):
            target_contract = resolved
        else:
            return make_call_node('X', [f"{declared_type}.{func_name}"])

    candidates = collect_call_candidates(target_contract, func_name, project_index)
    if not candidates:
        return make_call_node('X', [f"{target_contract['name']}.{func_name}"])
    kind = classify_kind(target_contract['kind'], candidates[0][1]['visibility'])
    targets = [f"{c['name']}.{f['name']}" for c, f in candidates]
    return make_call_node(kind, targets)


def resolve_call(chain: List[str], func_name: str, ctx: dict) -> Optional[dict]:
    """Implements spec Section 4, extended (deliberately, beyond the
    letter of Section 4.3) to also resolve receivers that are local
    variables/parameters, direct contract/library name qualifiers, and
    struct-field chains off any of those (`x.y.z.f()`) - not only bare
    state variables. `chain` is the dotted identifiers before the call
    name, base first, or [] for a bare call. Returns a node dict
    (I/X/L/UNRESOLVED_CALL), or None if `func_name` names no user-defined
    function and no call node applies (a builtin namespace) - argument
    reads are still scanned by the caller either way, so no state read is
    ever dropped."""
    contract = ctx['contract']
    project_index = ctx['project_index']

    if chain and func_name in LOW_LEVEL_CALLS:
        return {'type': 'UNRESOLVED_CALL', 'receiver': '.'.join(chain), 'kind': func_name}

    if not chain:
        candidates = collect_call_candidates(contract, func_name, project_index)
        if not candidates:
            return None
        kind = classify_kind(candidates[0][0]['kind'], candidates[0][1]['visibility'])
        targets = [f"{c['name']}.{f['name']}" for c, f in candidates]
        return make_call_node(kind, targets)

    if len(chain) == 1 and chain[0] == 'this':
        # `this.f()` always dispatches as an external call, regardless of
        # f's own declared visibility - a deterministic Solidity rule,
        # not a guess.
        candidates = collect_call_candidates(contract, func_name, project_index)
        if not candidates:
            return None
        targets = [f"{c['name']}.{f['name']}" for c, f in candidates]
        return make_call_node('X', targets)

    if chain[0] in BUILTIN_NAMESPACES:
        return None

    fallback_target = f"{'.'.join(chain)}.{func_name}"

    base = chain[0]
    base_type = ctx['state_var_types'].get(base)
    if base_type is None:
        base_type = _local_type(base, ctx['shadow_stack'])

    if base_type is None:
        if len(chain) == 1:
            # Not a variable at all - check whether `base` is itself the
            # literal name of a contract/library, i.e. a direct
            # static-qualified call (`LibraryName.func(...)`), which
            # needs no instance variable. This is how library calls are
            # written when not using `using ... for`.
            named_contract = project_index.resolve(base)
            if named_contract is not None and named_contract['kind'] in ('contract', 'abstract contract', 'library'):
                return _resolve_declared_type_call(base, func_name, project_index)
        # Truly unresolvable base (not a state var, not a tracked
        # local/param, not a known contract/library name) - still emit a
        # node rather than drop the edge, per Section 4.3 step 3's
        # "nameable but unpopulated" shape.
        return make_call_node('X', [fallback_target])

    current_type = base_type
    for field_name in chain[1:]:
        fields = _resolve_struct_fields(contract, current_type, project_index)
        if fields is None or field_name not in fields:
            # Can't walk further (unknown/non-struct type, or the field
            # isn't declared in it) - emit the full chain as a transparent
            # placeholder rather than a specific but unverified guess.
            return make_call_node('X', [fallback_target])
        current_type = fields[field_name]

    return _resolve_declared_type_call(current_type, func_name, project_index)


# ==========================================================================
# Expression scanning: state-variable reads + nested calls (spec 3.1, 5.2, 5.3)
# ==========================================================================

def _shadowed(name: str, shadow_stack: List[Dict[str, Optional[str]]]) -> bool:
    return any(name in scope for scope in shadow_stack)


def _backward_chain(text: str, pos: int) -> Tuple[List[str], int]:
    """Walk backward from `pos` (start of a call name) through zero or
    more `.identifier` segments. Returns (chain, base_start) where
    `chain` is the dotted identifiers before the call name, base first
    (empty if there's no preceding dot), and `base_start` is the absolute
    start offset of the leftmost identifier (== pos if chain is empty)."""
    segments: List[str] = []
    cur = pos
    while True:
        j = cur
        while j > 0 and text[j - 1] in ' \t\r\n':
            j -= 1
        if j == 0 or text[j - 1] != '.':
            break
        j -= 1
        while j > 0 and text[j - 1] in ' \t\r\n':
            j -= 1
        m = re.search(r'([A-Za-z_]\w*)$', text[:j])
        if not m:
            break
        segments.insert(0, m.group(1))
        cur = m.start(1)
    return segments, cur


def scan_expr(text: str, abs_start: int, ctx: dict) -> List[Tuple[int, dict]]:
    """Walks `text` left to right, emitting (abs_pos, node) for every state
    read (R) and every resolvable function call (I/X/L/UNRESOLVED_CALL),
    recursing into call arguments. `ctx` carries: contract, project_index,
    state_var_names (set), state_var_types (dict), shadow_stack (list of
    sets, innermost last)."""
    results: List[Tuple[int, dict]] = []
    n = len(text)
    consumed_until = 0
    for m in re.finditer(r'[A-Za-z_]\w*', text):
        start, end = m.start(), m.end()
        if start < consumed_until:
            continue
        word = m.group(0)

        j = skip_ws(text, end)
        if j < n and text[j] == '(':
            chain, chain_start = _backward_chain(text, start)
            close = find_matching_paren(text, j)
            args_text = text[j + 1:close]
            args_abs = abs_start + j + 1
            node = resolve_call(chain, word, ctx)
            if node is not None:
                results.append((abs_start + chain_start, node))
            results.extend(scan_expr(args_text, args_abs, ctx))
            consumed_until = close + 1
            continue

        prefix = text[:start].rstrip()
        if prefix.endswith('.'):
            consumed_until = end
            continue

        if j < n and text[j] == '.':
            j2 = j
            while j2 < n and text[j2] == '.':
                m3 = re.match(r'\.\s*[A-Za-z_]\w*', text[j2:])
                if not m3:
                    break
                j2 += m3.end()
                j2 = skip_ws(text, j2)
            if j2 > j and j2 < n and text[j2] == '(':
                consumed_until = end
                continue

        if not _shadowed(word, ctx['shadow_stack']) and word in ctx['state_var_names']:
            results.append((abs_start + start, {'type': 'R', 'target': _qualified_target(word, ctx)}))
        consumed_until = end

    return results


# ==========================================================================
# Statement classifier (spec Section 5)
# ==========================================================================

def try_local_decls(stmt_text: str) -> List[Tuple[str, Optional[str]]]:
    """Detects a local variable declaration (any kind, including tuple
    destructuring) and returns the newly declared (name, type) pair(s)
    (type is None where a single element can't be typed, e.g. reassigning
    into an existing tuple element). Reuses `parse_decl_name`'s
    `<type> <name>` grammar - safe against false positives on
    non-declaration statements because those fail its trailing
    `^\\w+$` name check."""
    stmt = strip_default_value(stmt_text).strip()
    if not stmt:
        return []
    if stmt.startswith('('):
        try:
            close = find_matching_paren(stmt, 0)
        except ValueError:
            return []
        out = []
        for part in stmt[1:close].split(','):
            p = part.strip()
            if not p:
                continue
            toks = p.split()
            if len(toks) >= 2 and re.match(r'^\w+$', toks[-1]):
                out.append((toks[-1], toks[0]))
        return out
    parsed = parse_decl_name(stmt)
    if parsed is None:
        return []
    name, type_token, _mods = parsed
    return [(name, type_token)]


REQUIRE_RE = re.compile(r'^(require|assert)\s*\(')
EMIT_RE = re.compile(r'^emit\s+(\w+)\s*\(')


def _split_first_top_level_comma(text: str) -> str:
    depth = 0
    for i, c in enumerate(text):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == ',' and depth == 0:
            return text[:i]
    return text


def classify_statement(text: str, abs_start: int, ctx: dict) -> List[Tuple[int, dict]]:
    """Returns [(abs_pos, node), ...] for one non-control statement, in
    left-to-right source order."""
    stripped = text.strip()
    if not stripped:
        return []
    lead = len(text) - len(text.lstrip())
    base_off = abs_start + lead

    alias_m = STORAGE_ALIAS_RE.match(stripped)
    if alias_m:
        alias_name, expr_text = alias_m.group(2), alias_m.group(3)
        ident = base_identifier(expr_text)
        if ident is not None:
            ctx['aliases'][alias_name] = _resolve_root(ident, ctx['aliases'])
        ctx['shadow_stack'][-1][alias_name] = alias_m.group(1)
        return []

    delete_m = re.match(r'delete\s+(.+)$', stripped, re.DOTALL)
    if delete_m:
        ident = base_identifier(delete_m.group(1))
        if ident is not None and not _shadowed(ident, ctx['shadow_stack']):
            root = _resolve_root(ident, ctx['aliases'])
            if root in ctx['state_var_names']:
                return [(base_off, {'type': 'W', 'target': _qualified_target(root, ctx)})]
        return []

    m = EMIT_RE.match(stripped)
    if m:
        return [(base_off, {'type': 'EMIT', 'event': m.group(1)})]

    m = REQUIRE_RE.match(stripped)
    if m:
        open_p = stripped.index('(', m.start())
        close_p = find_matching_paren(stripped, open_p)
        cond_text = _split_first_top_level_comma(stripped[open_p + 1:close_p])
        cond_abs = base_off + open_p + 1
        condition_reads = [n for _pos, n in scan_expr(cond_text, cond_abs, ctx) if n['type'] == 'R']
        node = {'type': 'REQUIRE', 'condition': cond_text.strip(), 'conditionReads': condition_reads}
        return [(base_off, node)]

    if re.match(r'^return\b', stripped):
        rest = stripped[len('return'):].strip()
        # Frozen schema: RETURN carries only the raw value text - its
        # expression is not scanned for embedded R/I/X/L nodes.
        return [(base_off, {'type': 'RETURN', 'value': rest if rest else None})]

    decls = try_local_decls(stripped)
    if decls:
        for n, t in decls:
            ctx['shadow_stack'][-1][n] = t
        eq = re.search(r'(?<![=!<>])=(?![=>])', stripped)
        if eq:
            rhs = stripped[eq.end():]
            rhs_abs = base_off + eq.end()
            return sorted(scan_expr(rhs, rhs_abs, ctx), key=lambda p: p[0])
        return []

    split = top_level_split_op(stripped, ASSIGN_OPS)
    if split:
        pos, _op = split
        lhs = stripped[:pos].strip()
        rhs = stripped[pos + 1:]
        rhs_abs = base_off + stripped.index(rhs, pos)
        out: List[Tuple[int, dict]] = []
        ident = base_identifier(lhs)
        if ident is not None and not _shadowed(ident, ctx['shadow_stack']):
            root = _resolve_root(ident, ctx['aliases'])
            if root in ctx['state_var_names']:
                out.append((base_off, {'type': 'W', 'target': _qualified_target(root, ctx)}))
        out.extend(scan_expr(rhs, rhs_abs, ctx))
        return sorted(out, key=lambda p: p[0])

    prefix_m = re.match(r'^(\+\+|--)(.+)$', stripped)
    postfix_m = re.match(r'^(.+?)(\+\+|--)$', stripped) if not prefix_m else None
    if prefix_m or postfix_m:
        target_text = prefix_m.group(2) if prefix_m else postfix_m.group(1)
        ident = base_identifier(target_text)
        if ident is not None and not _shadowed(ident, ctx['shadow_stack']):
            root = _resolve_root(ident, ctx['aliases'])
            if root in ctx['state_var_names']:
                return [(base_off, {'type': 'W', 'target': _qualified_target(root, ctx)})]
        return []

    if PUSH_POP_RE.match(stripped):
        ident = base_identifier(stripped)
        out = []
        if ident is not None and not _shadowed(ident, ctx['shadow_stack']):
            root = _resolve_root(ident, ctx['aliases'])
            if root in ctx['state_var_names']:
                out.append((base_off, {'type': 'W', 'target': _qualified_target(root, ctx)}))
        paren_idx = stripped.find('(')
        if paren_idx != -1:
            close_p = find_matching_paren(stripped, paren_idx)
            args_abs = base_off + paren_idx + 1
            out.extend(scan_expr(stripped[paren_idx + 1:close_p], args_abs, ctx))
        return sorted(out, key=lambda p: p[0])
        return []

    return sorted(scan_expr(stripped, base_off, ctx), key=lambda p: p[0])


def trace_walk_block(text: str, abs_start: int, ctx: dict) -> List[dict]:
    children: List[dict] = []
    i = 0
    n = len(text)
    while i < n:
        i = skip_ws(text, i)
        if i >= n:
            break
        if text[i] == ';':
            i += 1
            continue
        if text[i] == '{':
            close = find_matching_brace(text, i)
            ctx['shadow_stack'].append({})
            children.extend(trace_walk_block(text[i + 1:close], abs_start + i + 1, ctx))
            ctx['shadow_stack'].pop()
            i = close + 1
            continue
        if re.match(r'(if|for|while)\b', text[i:]):
            nodes, i = trace_handle_control(text, i, abs_start, ctx)
            children.extend(nodes)
            continue
        if re.match(r'unchecked\b', text[i:]):
            j = skip_ws(text, i + len('unchecked'))
            if j < n and text[j] == '{':
                close = find_matching_brace(text, j)
                ctx['shadow_stack'].append({})
                children.extend(trace_walk_block(text[j + 1:close], abs_start + j + 1, ctx))
                ctx['shadow_stack'].pop()
                i = close + 1
                continue
        if re.match(r'do\b', text[i:]):
            j = skip_ws(text, i + 2)
            if j < n and text[j] == '{':
                close = find_matching_brace(text, j)
                ctx['shadow_stack'].append({})
                children.extend(trace_walk_block(text[j + 1:close], abs_start + j + 1, ctx))
                ctx['shadow_stack'].pop()
                k = skip_ws(text, close + 1)
                if re.match(r'while\b', text[k:]):
                    wp = skip_ws(text, k + 5)
                    wclose = find_matching_paren(text, wp)
                    stmt_end = find_stmt_end(text, wclose + 1)
                    i = stmt_end + 1
                    continue
                i = close + 1
                continue
        stmt_end = find_stmt_end(text, i)
        for _pos, node in classify_statement(text[i:stmt_end], abs_start + i, ctx):
            children.append(node)
        i = stmt_end + 1
    return children


def trace_handle_control(text: str, i: int, abs_start: int, ctx: dict) -> Tuple[List[dict], int]:
    kw = re.match(r'(if|for|while)\b', text[i:]).group(1)
    j = skip_ws(text, i + len(kw))
    paren_open = j
    close_p = find_matching_paren(text, paren_open)
    cond = text[paren_open + 1:close_p]
    k = skip_ws(text, close_p + 1)

    ctx['shadow_stack'].append({})
    condition_reads: List[dict] = []
    if kw == 'for':
        # a for-loop's condition is the literal keyword "for" - its
        # init/test/increment clauses are not scanned for reads or calls.
        # The init clause is still walked (discarding any nodes it would
        # produce) purely so a loop variable it declares (e.g.
        # `uint256 i = 0`) is registered as a local and doesn't get
        # mistaken for a state variable inside the loop body.
        depth = 0
        start_idx = 0
        parts = []
        for idx, c in enumerate(cond):
            if c in '([{':
                depth += 1
            elif c in ')]}':
                depth -= 1
            if c == ';' and depth == 0:
                parts.append(cond[start_idx:idx])
                start_idx = idx + 1
        parts.append(cond[start_idx:])
        if len(parts) >= 1 and parts[0].strip():
            classify_statement(parts[0], abs_start + paren_open + 1, ctx)
        cond_text = 'for'
    else:
        cond_text = cond.strip() if kw == 'if' else kw
        if kw == 'if':
            # R nodes found anywhere in the condition (including inside
            # negations, boolean combinations, nested calls' arguments,
            # etc.) are nested inside the IF node, not scanned separately
            # per-operator - a single scan_expr pass over the whole
            # condition text already walks every sub-expression.
            condition_reads = [n for _pos, n in scan_expr(cond, abs_start + paren_open + 1, ctx)
                                if n['type'] == 'R']

    if k < len(text) and text[k] == '{':
        close_b = find_matching_brace(text, k)
        body_children = trace_walk_block(text[k + 1:close_b], abs_start + k + 1, ctx)
        after = close_b + 1
    else:
        stmt_end = find_stmt_end(text, k)
        body_children = [n for _p, n in classify_statement(text[k:stmt_end], abs_start + k, ctx)]
        after = stmt_end + 1
    ctx['shadow_stack'].pop()

    if kw == 'if':
        node = {'type': 'IF', 'condition': cond_text, 'conditionReads': condition_reads,
                'children': body_children}
        p = skip_ws(text, after)
        if text[p:p + 4] == 'else' and (p + 4 >= len(text) or not (text[p + 4].isalnum() or text[p + 4] == '_')):
            p2 = skip_ws(text, p + 4)
            ctx['shadow_stack'].append({})
            if re.match(r'if\b', text[p2:]):
                else_nodes, after = trace_handle_control(text, p2, abs_start, ctx)
                else_children = else_nodes
            elif p2 < len(text) and text[p2] == '{':
                close_b2 = find_matching_brace(text, p2)
                else_children = trace_walk_block(text[p2 + 1:close_b2], abs_start + p2 + 1, ctx)
                after = close_b2 + 1
            else:
                stmt_end2 = find_stmt_end(text, p2)
                else_children = [n for _p, n in classify_statement(text[p2:stmt_end2], abs_start + p2, ctx)]
                after = stmt_end2 + 1
            ctx['shadow_stack'].pop()
            node['children'] = node['children'] + [{'type': 'ELSE', 'children': else_children}]
        return [node], after

    node = {'type': 'LOOP', 'condition': cond_text, 'children': body_children}
    return [node], after


def _param_names(func: dict, contract) -> Dict[str, Optional[str]]:
    header_re = re.compile(r'\bfunction\s+' + re.escape(func['name']) + r'\s*\(')
    m = header_re.search(contract['body'])
    if m is None:
        return {}
    open_p = contract['body'].index('(', m.start())
    close_p = find_matching_paren(contract['body'], open_p)
    params_text = contract['body'][open_p + 1:close_p]
    names: Dict[str, Optional[str]] = {}
    for part in params_text.split(','):
        toks = part.strip().split()
        if len(toks) >= 2 and re.match(r'^\w+$', toks[-1]):
            names[toks[-1]] = toks[0]
    return names


def trace_function(contract: dict, func: dict, project_index: ProjectIndex) -> List[dict]:
    """Entry point for the trace-generation layer (spec Sections 3-5).
    Returns the ordered list of top-level TraceNodes for `func`'s body."""
    if func['body'] is None:
        return []

    # inheritance-aware, cross-file state-var collection (uses
    # project_index so bases declared in other files are included too).
    # Two passes: `mutable_only=True` (default) feeds W/R node detection,
    # unchanged from prior behavior. A second, wider pass also includes
    # constant/immutable declarations - irrelevant for W/R (nothing
    # "writes" an immutable post-construction) but required for resolving
    # a call receiver's declared type, since immutable address-typed vars
    # are a common way to hold a contract reference (e.g.
    # `ITroveNFT public immutable troveNFT`).
    def _collect_types(c, include_constants, seen=None):
        if seen is None:
            seen = set()
        if c['name'] in seen:
            return []
        seen.add(c['name'])
        out = []
        for base_name in c['bases']:
            base = project_index.resolve(base_name)
            if base is not None:
                out.extend(_collect_types(base, include_constants, seen))
        out.extend(extract_state_var_names(c['body'], include_constants))
        return out

    state_var_names = {name for name, _t in _collect_types(contract, False)}
    state_var_types = dict(_collect_types(contract, True))

    ctx = {
        'contract': contract,
        'project_index': project_index,
        'state_var_names': state_var_names,
        'state_var_types': state_var_types,
        'shadow_stack': [_param_names(func, contract)],
        'aliases': {},
        'owner_cache': {},
    }
    trace = trace_walk_block(mask_string_literals(func['body']), func['body_start_abs'], ctx)
    if not trace or trace[-1]['type'] != 'RETURN':
        trace.append({'type': 'RETURN', 'value': None})
    return trace


# ==========================================================================
# Output
# ==========================================================================

def run_trace_mode(project_root: str, spec: str, output_path: Optional[str]) -> None:
    import json

    m = EXTRACT_SPEC_RE.match(spec.strip())
    if not m:
        raise SystemExit(f"error: malformed --trace spec '{spec}' - expected Contract.functionName")
    contract_name, func_name = m.group(1), m.group(2)

    project_index = ProjectIndex.build(project_root)
    try:
        contract = project_index.resolve(contract_name)
    except ContractAmbiguityError as e:
        raise SystemExit(str(e))
    if contract is None:
        raise SystemExit(f"error: no contract/interface/library named '{contract_name}' under {project_root}")

    candidates = collect_call_candidates(contract, func_name, project_index)
    if not candidates:
        raise SystemExit(f"error: no function named '{func_name}' found on '{contract_name}'")

    results = []
    for c, f in candidates:
        results.append({
            'contract': c['name'],
            'function': f['name'],
            'trace': trace_function(c, f, project_index),
        })
    payload = results[0] if len(results) == 1 else {'candidates': results}
    output_text = json.dumps(payload, indent=2) + "\n"

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)
        except OSError as e:
            raise SystemExit(f"error: could not write to {output_path}: {e}")
        print(f"Wrote trace for {spec} to {output_path}")
    else:
        sys.stdout.write(output_text)


def _function_source(contract: dict, func: dict) -> str:
    return contract['raw'][func['header_start_abs']:func['end_abs']].strip('\n')


def _all_contract_records(project_index: ProjectIndex) -> List[dict]:
    out = []
    for entry in project_index.by_name.values():
        if isinstance(entry, list):
            out.extend(entry)
        else:
            out.append(entry)
    return out


def _inherits_from_or_is(contract: dict, target_name: str, project_index: ProjectIndex,
                          seen=None) -> bool:
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return False
    seen.add(contract['name'])
    if contract['name'] == target_name:
        return True
    for base_name in contract['bases']:
        base = project_index.resolve(base_name)
        if base is not None and _inherits_from_or_is(base, target_name, project_index, seen):
            return True
    return False


def find_writers_cross_project(declaring_contract_name: str, var_name: str,
                                project_index: ProjectIndex,
                                newline_cache: Dict[str, List[int]]) -> List[dict]:
    """Every write to `var_name` (declared in `declaring_contract_name`)
    across the whole project - `declaring_contract_name` itself and every
    contract, in any file, that inherits from it, since an inherited
    state variable is shared storage writable from any subclass."""
    writers = []
    for contract in _all_contract_records(project_index):
        if not _inherits_from_or_is(contract, declaring_contract_name, project_index):
            continue
        mods_by_name = extract_modifiers_from_body(contract['body'], contract['body_start'])
        for func in extract_functions_from_body(contract['body'], contract['body_start']):
            hit = scan_function_for_write(func, mods_by_name, var_name)
            if hit is None:
                continue
            offset, via_modifier = hit
            path = contract['file_path']
            if path not in newline_cache:
                newline_cache[path] = compute_newline_offsets(contract['raw'])
            writers.append({
                'contract': contract['name'],
                'function': func['name'],
                'viaModifier': via_modifier,
                'file': path,
                'line': line_number(newline_cache[path], offset),
                'ref': f"{contract['name']}.{func['name']}",
            })
    return writers


def _walk_trace_nodes(nodes: List[dict]):
    """Yields every node in a trace, recursing into IF/ELSE/LOOP children
    and IF/REQUIRE conditionReads."""
    for node in nodes:
        yield node
        if 'children' in node:
            yield from _walk_trace_nodes(node['children'])
        if 'conditionReads' in node:
            yield from node['conditionReads']


def build_investigation_package(project_root: str, root_spec: str) -> dict:
    """The recursion/closure layer: starting from one root function,
    transitively traces every function reachable via I/X/L call edges
    (each expanded exactly once, cycle-safe), and for every R (read)
    state variable encountered anywhere in that closure, attaches its
    project-wide writer list (each writer itself expanded into a full
    Function Document too, recursively). W (write) nodes are not
    cross-referenced - only reads can influence a function's own
    execution, so only reads get a "where else can this value change"
    lookup. Output follows a frozen schema (see project notes) -
    {metadata, rootFunction, functions, stateVariables} - and must never
    deviate from it."""
    m = EXTRACT_SPEC_RE.match(root_spec.strip())
    if not m:
        raise SystemExit(f"error: malformed spec '{root_spec}' - expected Contract.functionName")
    root_contract_name, root_func_name = m.group(1), m.group(2)

    project_index = ProjectIndex.build(project_root)
    try:
        root_contract = project_index.resolve(root_contract_name)
    except ContractAmbiguityError as e:
        raise SystemExit(str(e))
    if root_contract is None:
        raise SystemExit(f"error: no contract/interface/library named '{root_contract_name}' under {project_root}")

    functions: Dict[str, dict] = {}
    state_variables: Dict[str, dict] = {}
    newline_cache: Dict[str, List[int]] = {}
    queued = {f"{root_contract_name}.{root_func_name}"}
    pending = [(root_contract_name, root_func_name)]

    def enqueue(cname: str, fname: str) -> None:
        tkey = f"{cname}.{fname}"
        if tkey not in queued:
            queued.add(tkey)
            pending.append((cname, fname))

    def stub(key: str) -> dict:
        return {'id': key, 'trace': None, 'implementation': {'language': 'solidity', 'source': None}}

    while pending:
        contract_name, func_name = pending.pop(0)
        key = f"{contract_name}.{func_name}"
        if key in functions:
            continue

        try:
            contract = project_index.resolve(contract_name)
        except ContractAmbiguityError as e:
            raise SystemExit(str(e))
        if contract is None:
            functions[key] = stub(key)
            continue

        own_funcs = [f for f in extract_functions_from_body(contract['body'], contract['body_start'])
                     if f['name'] == func_name]
        implemented = [f for f in own_funcs if f['body'] is not None]
        chosen = implemented[0] if implemented else (own_funcs[0] if own_funcs else None)

        if chosen is None or chosen['body'] is None:
            functions[key] = stub(key)
            continue

        trace = trace_function(contract, chosen, project_index)
        functions[key] = {
            'id': key,
            'trace': trace,
            'implementation': {'language': 'solidity', 'source': _function_source(contract, chosen)},
        }

        for node in _walk_trace_nodes(trace):
            ntype = node['type']
            if ntype in ('I', 'X', 'L'):
                cname, _, fname = node['target'].partition('.')
                if not fname:
                    continue
                try:
                    named = project_index.resolve(cname)
                except ContractAmbiguityError:
                    named = None
                if named is None or named['kind'] not in ('contract', 'abstract contract', 'library'):
                    if node['target'] not in functions:
                        functions[node['target']] = stub(node['target'])  # leaf: not in this project
                    continue
                enqueue(cname, fname)
            elif ntype == 'R':
                skey = node['target']
                if skey in state_variables or '.' not in skey:
                    continue
                owner, _, var_name = skey.partition('.')
                writer_records = find_writers_cross_project(owner, var_name, project_index, newline_cache)
                for w in writer_records:
                    enqueue(w['contract'], w['function'])
                writer_ids = [f"{w['contract']}.{w['function']}" for w in writer_records]
                state_variables[skey] = {'id': skey, 'writers': writer_ids if writer_ids else None}

    return {
        'metadata': {'contract': root_contract_name},
        'rootFunction': f"{root_contract_name}.{root_func_name}",
        'functions': functions,
        'stateVariables': state_variables,
    }


def run_closure_mode(project_root: str, spec: str, output_path: Optional[str]) -> None:
    import json

    package = build_investigation_package(project_root, spec)
    output_text = json.dumps(package, indent=2) + "\n"

    path = output_path or "investigation_package.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(output_text)
    except OSError as e:
        raise SystemExit(f"error: could not write to {path}: {e}")
    print(f"Wrote investigation package for {spec} to {path}")


def print_ambiguous_error(project_root: str, variable: str) -> None:
    print("Error", file=sys.stderr)
    print(file=sys.stderr)
    print("State variable is ambiguous.", file=sys.stderr)
    print(file=sys.stderr)
    print("Please specify the owning contract.", file=sys.stderr)
    print(file=sys.stderr)
    print("Example", file=sys.stderr)
    print(file=sys.stderr)
    print(f"solindex {project_root} <ContractName>.sol {variable}", file=sys.stderr)


def _writer_impl_text(writer: dict, raw: str) -> str:
    """Renders the source text to show for one writer: the function's own
    signature+body verbatim (comments and formatting intact) when it has a
    body, or the bare signature plus the modifier's implementation when
    the write is only reachable through an applied modifier."""
    func = writer['func']
    header_start, end = func['header_start_abs'], func['end_abs']
    own_text = raw[header_start:end].strip('\n')

    if writer['matched_modifier'] is None:
        return own_text

    mod = writer['mod']
    if mod is None:
        return own_text

    mod_text = raw[mod['header_start_abs']:mod['end_abs']].strip('\n')
    if func['body'] is not None:
        # Function has its own body, but the actual write happened inside
        # a modifier applied to it - show both, the modifier is what to
        # look at for the write itself.
        return (f"{own_text}\n\n"
                f"// write occurs in modifier '{writer['matched_modifier']}':\n"
                f"{mod_text}")
    # Function itself is body-less (e.g. abstract signature) - the
    # modifier is the only place the write actually happens.
    return (f"{own_text}\n\n"
            f"// write occurs in modifier '{writer['matched_modifier']}':\n"
            f"{mod_text}")


def render_text(contract_name: str, variable: str, writers: List[dict],
                 raw: str, include_locations: bool, include_impl: bool) -> str:
    header = f"{contract_name}.{variable}"
    divider = "\u2500" * max(28, len(header))

    sections: List[str] = [f"State Variable\n{header}"]

    if not writers:
        sections.append(f"W({variable})\n\n(no writers found)")
        return f"\n{divider}\n".join(sections) + "\n"

    tree_lines = [f"W({variable})"]
    last_idx = len(writers) - 1
    for idx, w in enumerate(writers):
        is_last = idx == last_idx
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        tree_lines.append(f"{connector}{w['func']['name']}()")
        if include_locations and w['loc']:
            pad = "    " if is_last else "\u2502   "
            tree_lines.append(f"{pad}{w['loc']}")
    sections.append("\n".join(tree_lines))

    if include_impl:
        for w in writers:
            sections.append(_writer_impl_text(w, raw))

    return f"\n{divider}\n".join(sections) + "\n"


def render_markdown(contract_name: str, variable: str, writers: List[dict],
                     raw: str, include_locations: bool, include_impl: bool) -> str:
    header = f"{contract_name}.{variable}"
    lines: List[str] = [f"# State Variable: `{header}`", "", f"## W({variable})", ""]

    if not writers:
        lines.append("_(no writers found)_")
        return "\n".join(lines) + "\n"

    tree_lines = []
    last_idx = len(writers) - 1
    for idx, w in enumerate(writers):
        is_last = idx == last_idx
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        line = f"{connector}{w['func']['name']}()"
        if include_locations and w['loc']:
            line += f"   ({w['loc']})"
        tree_lines.append(line)
    lines.append("```")
    lines.extend(tree_lines)
    lines.append("```")

    if include_impl:
        lines.append("")
        lines.append("## Writer Implementations")
        for w in writers:
            lines.append("")
            lines.append(f"### `{w['func']['name']}()`")
            lines.append("")
            lines.append("```solidity")
            lines.append(_writer_impl_text(w, raw))
            lines.append("```")

    return "\n".join(lines) + "\n"


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="solindex",
        description="Enumerate every function capable of writing a given storage variable.",
    )
    parser.add_argument(
        "positional", nargs='+', metavar="ARGS",
        help="<project_root> <contract> <variable> - e.g. contracts/ ActivePool.sol collBalance. "
             "Omitting <contract> (i.e. passing only 2 arguments) triggers the ambiguous-query error.",
    )
    parser.add_argument(
        "--locations", action="store_true",
        help="Include a file:line source location under each writer (optional metadata).",
    )
    parser.add_argument(
        "--no-impl", action="store_true",
        help="Print only the W(...) tree and skip the full source implementation "
             "of each writer function (implementations are shown by default).",
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH",
        help="Write the result to PATH instead of printing to the terminal. "
             "If PATH ends in .md, Markdown formatting is used automatically "
             "(headings + fenced code blocks) unless --text overrides it.",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Force Markdown-formatted output (headings + fenced code blocks), "
             "even when printing to the terminal or writing to a non-.md file.",
    )
    parser.add_argument(
        "--md", action="store_true",
        help="Shorthand: write Markdown output straight to a file instead of "
             "the terminal. Auto-names the file '<Contract>.<variable>.md' in "
             "the current directory unless -o/--output is also given. "
             "Equivalent to --markdown -o <auto-name>.",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Force plain-text output (the default terminal format), even when "
             "writing to a .md file.",
    )
    parser.add_argument(
        "--trace", metavar="Contract.function",
        help="Mechanical trace-generation mode (spec: SOLINDEX_TRACE_EXTENSION_SPEC.md): "
             "build a ProjectIndex over <project_root> and print the JSON TraceNode list "
             "for one function's body. Only <project_root> is required as the positional "
             "argument, e.g. solindex contracts/ --trace TroveManager.liquidate.",
    )
    parser.add_argument(
        "--closure", metavar="Contract.function",
        help="Recursive investigation-package mode: starting from one root function, "
             "transitively traces every function reachable via I/X/L call edges (each "
             "expanded once, cycle-safe), extracts each one's source, and attaches the "
             "project-wide writer list for every state variable READ anywhere in the "
             "closure. Prints one self-contained JSON document. Only <project_root> is "
             "required as the positional argument, e.g. "
             "solindex . --closure BorrowerOperations.closeTrove.",
    )
    parser.add_argument(
        "--extract", nargs='+', metavar="Contract.function",
        help="Independent function-extraction mode (unrelated to writer-indexing): "
             "pull one or more functions by 'Contract.functionName' and write their "
             "full source to a Markdown file (auto-named 'extracted_functions.md' "
             "unless -o/--output is given). Accepts multiple specs, e.g. "
             "--extract Vault.deposit Vault.mint Pool.withdraw. When --extract is "
             "used, only <project_root> is required as the positional argument.",
    )
    args = parser.parse_args()

    if args.trace:
        if len(args.positional) != 1:
            parser.error(
                "--trace mode takes exactly one positional argument: <project_root> "
                "(e.g. solindex contracts/ --trace Vault.deposit)"
            )
        project_root = args.positional[0]
        if not os.path.isdir(project_root):
            raise SystemExit(f"error: project root not found or not a directory: {project_root}")
        run_trace_mode(project_root, args.trace, args.output)
        return

    if args.closure:
        if len(args.positional) != 1:
            parser.error(
                "--closure mode takes exactly one positional argument: <project_root> "
                "(e.g. solindex . --closure BorrowerOperations.closeTrove)"
            )
        project_root = args.positional[0]
        if not os.path.isdir(project_root):
            raise SystemExit(f"error: project root not found or not a directory: {project_root}")
        run_closure_mode(project_root, args.closure, args.output)
        return

    if args.extract:
        # Function-extraction mode is fully independent of writer-indexing:
        # it needs only <project_root> and never touches the code paths
        # below (contract/variable resolution, writer scanning, etc).
        if len(args.positional) != 1:
            parser.error(
                "--extract mode takes exactly one positional argument: <project_root> "
                "(e.g. solindex contracts/ --extract Vault.deposit)"
            )
        project_root = args.positional[0]
        if not os.path.isdir(project_root):
            raise SystemExit(f"error: project root not found or not a directory: {project_root}")
        run_extract_mode(project_root, args.extract, args.output)
        return

    if args.md and args.text:
        parser.error("--md and --text are mutually exclusive")

    if len(args.positional) == 2:
        project_root, variable = args.positional
        print_ambiguous_error(project_root, variable)
        sys.exit(1)

    if len(args.positional) != 3:
        parser.error(
            "expected <project_root> <contract> <variable> "
            "(3 arguments; pass only 2 to see the ambiguous-query error)"
        )

    project_root, contract_arg, variable = args.positional

    if not os.path.isdir(project_root):
        raise SystemExit(f"error: project root not found or not a directory: {project_root}")

    path, raw, src, contracts_by_name, target_name = resolve_target(project_root, contract_arg)
    contract = contracts_by_name[target_name]

    state_var_names = {name for name, _type in collect_state_var_names(contract, contracts_by_name)}
    if variable not in state_var_names:
        raise SystemExit(
            f"error: state variable '{variable}' not found on contract '{target_name}' in {path} "
            f"(checked own and in-file inherited state variables; constants/immutables are excluded "
            f"since they are never storage-backed)"
        )

    func_order, mods_by_name = collect_functions_and_modifiers(contract, contracts_by_name)
    newline_offsets = compute_newline_offsets(src)

    writers: List[dict] = []
    for func in func_order:
        result = scan_function_for_write(func, mods_by_name, variable)
        if result is None:
            continue
        hit, matched_modifier = result
        loc = f"{os.path.relpath(path)}:{line_number(newline_offsets, hit)}"
        writers.append({
            'func': func,
            'loc': loc,
            'matched_modifier': matched_modifier,
            'mod': mods_by_name.get(matched_modifier) if matched_modifier else None,
        })

    if args.markdown and args.text:
        parser.error("--markdown and --text are mutually exclusive")

    output_path = args.output
    if args.md and not output_path:
        output_path = f"{target_name}.{variable}.md"

    use_markdown = args.markdown or args.md or (
        output_path and output_path.lower().endswith(".md") and not args.text
    )

    render = render_markdown if use_markdown else render_text
    output_text = render(target_name, variable, writers, raw,
                          args.locations, include_impl=not args.no_impl)

    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)
        except OSError as e:
            raise SystemExit(f"error: could not write to {output_path}: {e}")
        print(f"Wrote {'Markdown' if use_markdown else 'text'} output to {output_path}")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
