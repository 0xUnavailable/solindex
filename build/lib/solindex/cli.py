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


def parse_decl_name(stmt: str) -> Optional[Tuple[str, List[str]]]:
    """Parse a single declaration statement (no trailing ';') and return
    (name, modifiers), or None if it cannot be parsed as a
    `<type> <modifiers...> <name>` declaration. The type itself is
    discarded - solindex only needs the declared NAME to build the state
    variable name set."""
    stmt = strip_default_value(stmt)
    stmt = stmt.strip()
    if not stmt:
        return None

    if stmt.startswith('mapping'):
        paren_idx = stmt.find('(')
        if paren_idx == -1:
            return None
        close_idx = find_matching_paren(stmt, paren_idx)
        rest = stmt[close_idx + 1:].strip()
        rest_tokens = rest.split()
        if not rest_tokens:
            return None
        name = rest_tokens[-1]
        modifiers = rest_tokens[:-1]
        return (name, modifiers)

    tokens = stmt.split()
    if len(tokens) < 2:
        return None

    if tokens[0] == 'address' and len(tokens) >= 3 and tokens[1] == 'payable':
        rest_tokens = tokens[2:]
    else:
        rest_tokens = tokens[1:]

    if not rest_tokens:
        return None

    name = rest_tokens[-1]
    if not re.match(r'^\w+$', name):
        return None
    modifiers = rest_tokens[:-1]
    return (name, modifiers)


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


def extract_state_var_names(body: str) -> List[str]:
    """Given a contract body, strip function/modifier/struct/enum blocks,
    then parse remaining top-level statements for state variable
    declarations. Returns ordered declared names, skipping constants and
    immutables (never storage-backed, so never "writable" in the sense
    this tool cares about)."""
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
        name, modifiers = parsed
        if 'constant' in modifiers or 'immutable' in modifiers:
            continue
        names.append(name)
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


def collect_state_var_names(contract, contracts_by_name, seen=None) -> List[str]:
    """Recursively resolve in-file base contracts (declaration order),
    then this contract's own state variables. Bases not defined in this
    file are out of scope (single-file tool)."""
    if seen is None:
        seen = set()
    if contract['name'] in seen:
        return []
    seen.add(contract['name'])

    names: List[str] = []
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
        modifiers = []
        for mm in re.finditer(r'\b(\w+)\s*(\(([^)]*)\))?', htail):
            wname = mm.group(1)
            if not wname or wname in SKIP_HDR_WORDS:
                continue
            modifiers.append((wname, mm.group(3) or ''))

        funcs.append({
            'name': name, 'modifiers': modifiers,
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
# Output
# ==========================================================================

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
    args = parser.parse_args()

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

    state_var_names = set(collect_state_var_names(contract, contracts_by_name))
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
