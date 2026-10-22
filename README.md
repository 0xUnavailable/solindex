# solindex

Finds every function that writes a Solidity state variable, and traces what any function reads, writes, and calls, down through its full call graph.

solindex works directly on Solidity source text (comment/string-stripped, brace-matched), with no AST and no dependencies. Python 3.8+, standard library only.

## Install

```
cp cli.py solindex && chmod +x solindex
```

Or run `python3 cli.py ...` directly.

## Usage

`<project_root>` is scanned recursively for every `.sol` file.

```
# Who writes this state variable? (default mode)
solindex ./contracts ActivePool collBalance

# Exact source of specific functions
solindex ./contracts --extract Vault.deposit ActivePool.mintAggInterest

# Ordered reads, writes, calls, events and branches of one function
solindex . --trace BorrowerOperations.closeTrove

# Everything a function touches, transitively
solindex . --closure BorrowerOperations.closeTrove
```

| Mode | Output |
|------|--------|
| Writer-index (default) | Every function that writes a state variable, with source |
| `--extract` | Markdown with verbatim function source (default `extracted_functions.md`) |
| `--trace` | JSON array of TraceNodes in source order |
| `--closure` | JSON investigation package: call graph, sources, and writers, expanded recursively (default `investigation_package.json`) |

### Writer-index options

| Flag | Effect |
|------|--------|
| `--no-impl` | Only the `W(...)` tree, no source |
| `--locations` | Add `file:line` per writer |
| `--md` | Write to `<Contract>.<variable>.md` |
| `--markdown` | Force Markdown to stdout or `-o` |
| `-o PATH` | Write to a file |

```
$ solindex contracts Vault collBalance --no-impl
Vault.collBalance
W(collBalance)
├── deposit()
└── _settle()
```

`<contract>` can be a bare name, a filename, or a path. If a name is declared in more than one file, you'll get an error listing the candidates; pass a path to disambiguate. Inheritance is resolved across files.

## TraceNodes

Every node has a `type`. Fields are always present; a missing value is `null` or `[]`.

| Type | Fields | Meaning |
|------|--------|---------|
| `R` / `W` | `target` | State read / write, as `Contract.var` (declaring contract) |
| `I` | `target` | Internal/private call |
| `X` | `target` | External/public call, including `instance.f()` and `this.f()` |
| `L` | `target` | Library call |
| `UNRESOLVED_CALL` | `receiver`, `kind` | Low-level `.call`/`.delegatecall`/`.staticcall` (never resolved) |
| `EMIT` | `event` | Event emission (arguments not scanned) |
| `REQUIRE` | `condition`, `conditionReads` | `require`/`assert` guard |
| `IF` / `ELSE` | `condition`, `conditionReads`, `children` | Branch; trailing `ELSE` is inside `children` |
| `LOOP` | `condition`, `children` | `for`/`while`; only the body is walked |
| `RETURN` | `value` | Raw expression or `null`; always the last node |

Calls resolve through cross-file inheritance, interface-to-implementation, library-qualified calls, and struct-field-typed receivers. Unresolvable receivers still emit a node with a literal target.

## Closure schema

Frozen, so downstream tools can rely on it:

```
{
  "metadata": { "contract": "..." },
  "rootFunction": "Contract.function",
  "functions": {
    "Contract.function": {
      "id": "...",
      "trace": [ ...TraceNodes ] | null,
      "implementation": { "language": "solidity", "source": "..." | null }
    }
  },
  "stateVariables": {
    "Contract.var": { "id": "...", "writers": ["Contract.function", ...] | null }
  }
}
```

Every `I`/`X`/`L` target is a key in `functions`; every `R` target is a key in `stateVariables`; every writer is a key in `functions`. Functions outside the project appear with `trace: null` and `source: null`. Closure is cycle-safe, and each function is expanded once. Writes (`W`) are not cross-referenced.

## Limitations

- Receiver chains through brackets or nested calls (`address(this).call(...)`, `arr[i].foo()`) aren't resolved.
- `RETURN` values and loop headers aren't scanned for reads.
- Overloaded functions resolve to the first function with a body.

## License

MIT
