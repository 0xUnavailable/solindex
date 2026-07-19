# solindex

**A deterministic, non-LLM static analysis tool for Solidity.**

solindex answers questions about a Solidity codebase the way a compiler pass
would, not the way a language model would: no inference, no summarization,
no probabilistic guessing. Every answer is either derived mechanically from
the source text, or explicitly marked as unresolved. Given the same
codebase, solindex produces the same output, every time.

It does four things:

| Mode | Question it answers |
|---|---|
| **Writer-index** *(default)* | "Every function that can write to this state variable — where are they?" |
| `--extract` | "Give me the exact source of these functions." |
| `--trace` | "Walk this function's body and tell me every state read, state write, call, event, and branch, in order." |
| `--closure` | "Starting from this function, give me everything: full call graph, every function's source, and every state variable's writers — recursively, until there's nothing left to expand." |

Nothing in solindex parses Solidity into an AST. It works directly on
source text — comment/string-stripped, brace-matched, statement-classified
— which keeps it fast, dependency-free, and fully auditable: every rule it
follows is legible in the code that implements it.

---

## Requirements

- Python 3.8+
- No third-party dependencies. Standard library only.

## Installation

```bash
# Just the script - no packaging step
cp cli.py solindex
chmod +x solindex
./solindex --help
```

or invoke it directly with `python3 cli.py ...` as shown throughout this
document.

---

## Quick start

```bash
# Who writes ActivePool.collBalance, anywhere in the project?
solindex ./contracts ActivePool collBalance

# Give me the exact source of these two functions
solindex ./contracts --extract Vault.deposit ActivePool.mintAggInterest

# Walk BorrowerOperations.closeTrove() statement by statement
solindex . --trace BorrowerOperations.closeTrove

# Give me EVERYTHING closeTrove touches, transitively
solindex . --closure BorrowerOperations.closeTrove
```

`<project_root>` is scanned recursively — every `.sol` file under it, in
every subdirectory, is indexed in one pass. There's no need to flatten a
nested repo into a single folder; nested import structures resolve exactly
as Solidity itself would resolve them.

---

## Concepts

Two things solindex resolves before answering anything:

**Contract identity.** A contract/interface/library is identified by name,
project-wide. If the same name is declared in two different files, solindex
doesn't error immediately — indexing always succeeds. The error only fires
if you actually ask a question that requires resolving that specific name,
at which point solindex lists every file that declares it and asks you to
disambiguate (pass a full path instead of a bare name).

**Inheritance.** Base contracts are resolved across files, not just within
the same file as the contract that inherits them. A state variable declared
in `Base.sol` and written from `SubA.sol` and `SubB.sol` is discovered
correctly regardless of which file you start from.

---

## Mode 1: Writer-index *(default)*

```
solindex <project_root> <contract> <variable> [options]
```

`<contract>` accepts a bare contract name (searched project-wide), a
filename (`ActivePool.sol`), or a path. Passing only two arguments
(`<project_root> <variable>`) deliberately triggers the ambiguous-query
error, showing you the exact three-argument form to use.

```bash
$ solindex contracts Vault collBalance
State Variable
Vault.collBalance
────────────────────────────
W(collBalance)
├── deposit()
└── _settle()
────────────────────────────
function deposit(uint256 amount) external onlyOwner {
        require(amount > 0, "bad amount");
        collBalance += amount;
        ...
        _settle();
    }
────────────────────────────
function _settle() internal {
        collBalance = collBalance;
    }
```

Options:

| Flag | Effect |
|---|---|
| `--no-impl` | Print only the `W(...)` tree — skip full source of each writer |
| `--locations` | Add a `file:line` under each writer |
| `--md` | Write Markdown straight to `<Contract>.<variable>.md` |
| `--markdown` | Force Markdown formatting to stdout or `-o` |
| `-o PATH` | Write to a file (Markdown auto-detected from `.md` extension) |

```bash
$ solindex contracts Vault collBalance --no-impl
State Variable
Vault.collBalance
────────────────────────────
W(collBalance)
├── deposit()
└── _settle()
```

---

## Mode 2: `--extract`

Pull one or more functions verbatim, by `Contract.functionName`, into a
Markdown document. Independent of writer-indexing — doesn't require a
variable at all.

```bash
solindex contracts --extract Vault.deposit ActivePool.mintAggInterest -o funcs.md
```

Writes `# Extracted Functions`, one `## Contract.function` section per
spec, each with its source file and a fenced Solidity code block. Defaults
to `extracted_functions.md` if `-o` is omitted.

---

## Mode 3: `--trace`

```
solindex <project_root> --trace Contract.function [-o out.json]
```

Walks one function's body top to bottom and emits an ordered JSON array of
**TraceNodes** — every state read, state write, call, event, branch, and
return, in source order. This is the mechanical core the other modes are
built on.

```bash
$ solindex contracts --trace Vault.deposit
{
  "contract": "Vault",
  "function": "deposit",
  "trace": [
    { "type": "REQUIRE", "condition": "amount > 0", "conditionReads": [] },
    { "type": "W", "target": "Vault.collBalance" },
    { "type": "X", "target": "ActivePool.mintAggInterest" },
    {
      "type": "IF",
      "condition": "collBalance > 100",
      "conditionReads": [{ "type": "R", "target": "Vault.collBalance" }],
      "children": [
        { "type": "EMIT", "event": "Deposited" },
        { "type": "ELSE", "children": [{ "type": "W", "target": "Vault.collBalance" }] }
      ]
    },
    {
      "type": "LOOP",
      "condition": "for",
      "children": [{ "type": "W", "target": "Vault.collBalance" }]
    },
    { "type": "I", "target": "Vault._settle" },
    { "type": "RETURN", "value": null }
  ]
}
```

### TraceNode reference

Every node has a `type`. Fields are always present for that type — a
missing signal is `null`/`[]`, never an omitted key.

| Type | Fields | Meaning |
|---|---|---|
| `R` | `target` | A read of state variable `"Contract.varName"` (fully qualified by its *declaring* contract, resolved through inheritance). |
| `W` | `target` | A write to a state variable, same qualification. |
| `I` | `target` | An **internal**/private call — `"Contract.function"`. |
| `X` | `target` | An **external**/public call, including all `instanceVar.f()` and `this.f()` calls (Solidity dispatches these externally regardless of the callee's own declared visibility). |
| `L` | `target` | A call into a `library`. |
| `UNRESOLVED_CALL` | `receiver`, `kind` | A low-level `.call`/`.delegatecall`/`.staticcall` — deliberately never resolved to a target, since the destination is dynamic. |
| `EMIT` | `event` | An event emission. Arguments are not scanned. |
| `REQUIRE` | `condition`, `conditionReads` | A `require`/`assert` guard. `conditionReads` is the `R` nodes found inside the condition, nested here (never as external siblings). |
| `IF` | `condition`, `conditionReads`, `children` | A branch. `children` includes a trailing `ELSE` node when one exists. |
| `ELSE` | `children` | The else-branch of the preceding `IF`. |
| `LOOP` | `condition`, `children` | A `for`/`while` loop. `condition` is always the literal keyword (`"for"`/`"while"`) — the loop header's own test/init/increment clauses are not scanned for reads; only the loop **body** is walked. |
| `RETURN` | `value` | The raw return expression text, or `null` for a bare `return;`. **Not** scanned for embedded reads/calls — if you need to know what a returned value depends on, look at the reads/writes earlier in the same trace. Every function trace ends with a `RETURN` node — synthesized if the function doesn't already end in an explicit one. |

Call resolution (`I`/`X`/`L`) follows the real Solidity dispatch rules:
cross-file inheritance, interface-to-implementation resolution (`IFoo` →
concrete `Foo`), direct library-qualified calls (`Math.max(...)`), and
struct-field-typed receivers (`vars.pool.foo()`, resolved through the
struct's real declared field type, not a name-matching guess). When a
receiver genuinely can't be resolved, the node still exists — with a
transparent literal target — rather than being dropped.

---

## Mode 4: `--closure`

```
solindex <project_root> --closure Contract.function [-o investigation_package.json]
```

This is `--trace`, recursively. Starting from one root function, it:

1. Traces the root.
2. Expands every `I`/`X`/`L` target it finds into its own full trace —
   transitively, following the real call graph, not just one level deep.
3. For every `R` (read) it finds anywhere in that closure, looks up every
   function project-wide that can write that variable — cross-file,
   inheritance-aware — and expands **each writer** into its own full trace
   too, exactly like a call target. `W` (write) nodes are not
   cross-referenced: a write is an effect, not an input to execution.
4. Repeats until nothing new is left to expand.

Cycle-safe: mutual/indirect recursion terminates cleanly, and any
`Contract.function` is expanded exactly once no matter how many paths lead
to it. Defaults to writing `investigation_package.json` in the current
directory (override with `-o`).

```bash
$ solindex contracts --closure Vault.deposit
Wrote investigation package for Vault.deposit to investigation_package.json
```

### Investigation package schema

This schema is **frozen** — every `--closure` run produces exactly this
shape, regardless of the function or codebase, so downstream tooling can
parse it without special-casing.

```
{
  "metadata": { "contract": "<root contract name>" },
  "rootFunction": "Contract.function",

  "functions": {
    "Contract.function": {
      "id": "Contract.function",
      "trace": [ ...TraceNode... ] | null,
      "implementation": {
        "language": "solidity",
        "source": "<verbatim function source>" | null
      }
    },
    ...
  },

  "stateVariables": {
    "Contract.varName": {
      "id": "Contract.varName",
      "writers": ["Contract.function", ...] | null
    },
    ...
  }
}
```

Navigation rules — this is what "recursive" means in practice:

1. Opening a function always shows its **trace** and **implementation**
   together (`functions[id]`).
2. An `R(Contract.varName)` node anywhere in a trace points to
   `stateVariables["Contract.varName"]`.
3. Each entry in that variable's `writers` array is itself a key into
   `functions` — a full Function Document, not a stub.
4. Every `I`/`X`/`L` node's `target` is likewise a key into `functions`.
5. Repeat until a target has no corresponding entry (a truly external
   function — outside the indexed project — represented as
   `{"trace": null, "implementation": {"source": null}}`, never silently
   omitted) or you've reached a leaf with no further calls or reads.

```bash
$ python3 -c "
import json
pkg = json.load(open('investigation_package.json'))
fn = pkg['functions'][pkg['rootFunction']]
print(fn['implementation']['source'])
for node in fn['trace']:
    if node['type'] == 'R':
        print('reads', node['target'], '- writers:',
              pkg['stateVariables'][node['target']]['writers'])
"
```

---

## Design notes

- **No AST.** solindex operates on comment/string-masked source text with
  brace/paren matching and statement classification. This is a deliberate
  tradeoff: it means solindex has zero dependencies and its behavior is
  fully readable in its own source, at the cost of not handling every
  syntactic corner of Solidity (see below).
- **Never guess, never silently drop.** Where a call target or a read
  can't be resolved with certainty, solindex still emits a node — with an
  explicit, literal, honestly-unresolved target — rather than fabricating
  a plausible-looking answer or omitting the signal entirely.
- **Ambiguity is lazy.** Indexing a project with colliding contract names
  never fails on its own; it only fails when you ask a question whose
  answer depends on the colliding name, and at that point you get every
  candidate file listed so you can disambiguate.

### Known limitations

- Receiver chains through parenthesized expressions
  (`address(this).call(...)`, `arr[i].foo()`) aren't walked — the chain
  resolver handles plain `identifier.identifier.identifier(...)` chains,
  including struct-field types, but breaks at a bracket or nested call.
- `RETURN` values and loop headers (`for`/`while`) are not scanned for
  embedded state reads — see the TraceNode table above. If a function's
  *only* reference to a variable is inside a bare `return expr;` or a loop
  header, that reference won't appear as an `R` node.
- Overloaded functions resolve to the first implemented (bodied) match;
  the JSON schema carries a single `target` string, not a candidate list.
