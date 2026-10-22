# supertonic-py engineering handoff

Review date: **22 October 2026**. All findings remain **OPEN**. This handoff contains proposed repairs and release checks; no runtime source fixes or repaired-behavior acceptance are claimed.

**Reviewed source:** [`main` at `908a56486e821e833a80530ff0cae3ad0b046fce`](https://github.com/wwongw21/supertonic-py/tree/908a56486e821e833a80530ff0cae3ad0b046fce).

The full structured handoff is in [report.json](report.json). The historical commands and results below retain their original `/workspace` paths and are not portable reruns. Use the checked-in [standard-library reproducer](diagnostics/repro_pinned_sources.py) from the repository root; its fresh output is [captured here](evidence/portable-probe-2026-10-22.txt). It demonstrates pre-fix behavior only and does not apply runtime repairs.

## Audience and required outcome

### Audience

1. Maintainer implementing corrections

2. Reviewer checking behavior, preservation of user data and release readiness

### Outcome

null

## Scope and reviewed branches

Python SDK engine, text utilities, model/voice loader, CLI, FastAPI routes/schemas/style storage, dependency metadata, test and publish workflow configuration. No AGENTS.md present.

## Architecture

### Runtime flow

SDK/CLI TTS -> pipeline.TTS -> loader model/config/voice assets -> core.Supertonic ONNX sessions. Optional HTTP app holds one TTS instance, resolves built-in or custom style, serializes synthesis with a lock, and encodes WAV/FLAC/OGG.

### Components

#### Item 1

##### Component

Configuration and model identity

##### Files

1. supertonic/config.py

##### Role

Model names, per-model cache roots, pinned Hugging Face revisions, language/synthesis bounds and asset paths.

#### Item 2

##### Component

Download and loading

##### Files

1. supertonic/loader.py

##### Role

Snapshot download staging, model directory replacement, ONNX sessions, voice JSON loading and built-in voice lookup.

#### Item 3

##### Component

Synthesis

##### Files

1. supertonic/core.py

2. supertonic/pipeline.py

3. supertonic/utils.py

##### Role

Normalization, style tensors, duration/text/vector/vocoder inference and paragraph/sentence chunking with silence concatenation.

#### Item 4

##### Component

HTTP lifetime and body boundary

##### Files

1. supertonic/server/app.py

2. supertonic/server/schemas.py

##### Role

FastAPI lifetime loads model and styles; state owns synth lock; middleware prechecks import Content-Length; Pydantic describes requests.

#### Item 5

##### Component

HTTP operations and user styles

##### Files

1. supertonic/server/routes.py

2. supertonic/server/styles_store.py

3. supertonic/server/audio.py

##### Role

Voice resolution, native/OpenAI-compatible and batch synthesis, style persistence, audio encoding and errors.

#### Item 6

##### Component

CLI and release checks

##### Files

1. supertonic/cli.py

2. pyproject.toml

3. .github/workflows/test.yml

4. .github/workflows/publish.yml

##### Role

Public command parsing, optional serve extra, packaging and declared CI matrix.

### Contracts to preserve

1. SDK waveform shape (1, samples), floating waveform type and duration array return contract.

2. CLI defaults and model-specific cache behavior; SUPERTONIC_CACHE_DIR and SUPERTONIC_CUSTOM_STYLES_DIR overrides.

3. Built-in voice names remain reserved; user voice names remain sanitized.

4. OpenAI-shaped HTTP errors and supported WAV/FLAC/OGG content types.

5. Only optional [serve] installation should require FastAPI/uvicorn/multipart.

6. No modification of model revisions or weights is needed for these repairs.

## Implementation sequence

### Item 1

#### Order

1

#### Findings

1. SPY-1

#### Depends on

None recorded.

#### Deliverable

A user-data preservation regression plus downloader replacement that leaves imported voices and unmanaged content intact through success/failure. Fix data-loss risk first.

### Item 2

#### Order

2

#### Findings

1. SPY-2

#### Depends on

None recorded.

#### Deliverable

Schema omission semantics and explicit item/default/fallback precedence with route-level assertions.

### Item 3

#### Order

3

#### Findings

1. SPY-3

#### Depends on

None recorded.

#### Deliverable

Bound actual incoming import bytes before FastAPI parsing, preserving error shape and normal uploads.

### Item 4

#### Order

4

#### Findings

1. SPY-4

#### Depends on

None recorded.

#### Deliverable

Deterministic bounded chunking tests and Unicode-aware fallback splitting.

### Item 5

#### Order

5

#### Findings

1. SPY-1

2. SPY-2

3. SPY-3

4. SPY-4

#### Depends on

1. 1

2. 2

3. 3

4. 4

#### Deliverable

Rerun the offline selection, add repair regressions, then perform real model/CLI/server and Python/platform release acceptance. No acceptance in this step has run.

## Findings and implementation handoff

### SPY-1 High: Model re-download permanently removes imported custom voices

**Status:** OPEN. [Pinned source](https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/loader.py#L127).

#### Evidence classification

Confirmed through the validation described below

#### Observed mechanism

download_model deletes the entire existing model_dir with shutil.rmtree before moving the downloaded snapshot. The new server stores imported user styles at get_cache_dir(model)/custom_styles by default (server/styles_store.py:58). CLI cmd_download invokes this unconditionally (cli.py:222-223).

#### Impact and scope

After importing a custom voice, running the documented supertonic download command (or auto-repair after any ONNX module is missing) deletes those user voice files without warning or recovery. Even user-supplied model_dir contents are deleted. External originals/backups remain intact; a SUPERTONIC_CUSTOM_STYLES_DIR outside the model/cache directory avoids this deletion.

#### Repair direction

Keep user-imported styles outside the replaceable downloaded model directory, or preserve and restore the custom_styles directory while atomically replacing only managed model assets. Retain a backup until replacement succeeds.

#### Verification already obtained

Confirmed in a TemporaryDirectory using real loader.download_model and mocked huggingface_hub.snapshot_download: seeded custom_styles/my_voice.json existed before the call and was absent afterwards. No live user data touched. Reproducer: /workspace/review/repro_supertonic.py. Independently confirmed via the real CLI download and auto-download repair paths with temporary valid custom-style storage; only model initialization/HuggingFace download were mocked.

#### Locations

##### Item 1

###### File

supertonic/loader.py

###### Lines

1. 127

2. 131

###### Url

<https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/loader.py#L127>

#### Change surface

1. supertonic/loader.py: download_model and load_model

2. supertonic/server/styles_store.py: default_custom_styles_dir if storage ownership changes

3. supertonic/cli.py: cmd_download

4. tests/test_loader.py and tests/test_server_routes.py

5. Custom-style/download documentation if paths or recovery semantics change

#### Repair steps

1. Add a temporary-directory regression that imports a valid custom style through the real app, downloads a fake model snapshot and asserts the custom file is byte-identical and discoverable afterwards.

2. Choose explicit ownership: downloaded model assets may be replaced; custom_styles and other user-managed files must survive. The smallest compatible repair preserves user files when publishing a staged snapshot, without changing the public style path.

3. Prepare and validate the downloaded snapshot before retiring the previous usable model. Preserve custom_styles and unmanaged content; reserve model-managed destinations explicitly.

4. Publish using a recoverable directory swap/backup strategy rather than deleting the only existing tree first. Keep the old tree until publication and user-file preservation succeed; failures must restore it.

5. Use a unique staging directory for each operation and prevent competing downloads from deleting or replacing each other's temporary state.

6. If choosing a separate permanent user-data root instead, migrate existing per-model custom_styles transactionally and honor the explicit custom directory override. Avoid a silent path change that leaves imported styles undiscoverable.

#### Invariants

1. Successful model refresh preserves every imported user voice byte-for-byte.

2. A failed download, preservation copy or final publish leaves the previous model and user data accessible.

3. Model-specific voice isolation and explicit cache/custom-style overrides continue to work.

4. Managed model replacement must not recursively delete arbitrary unrelated files in a caller-provided model_dir.

#### Acceptance criteria

1. CLI redownload and missing-asset auto-download both retain existing custom styles and serve them after restart.

2. Failure injected before/after snapshot preparation and before final publication leaves original styles and existing usable model intact.

3. A custom styles directory outside the cache is untouched.

4. Concurrent refresh attempts produce one complete usable snapshot and no missing user files; if locking/concurrency is explicitly out of scope, return a clear conflict rather than corrupting state.

#### Regression scenarios

1. First download into missing directory

2. Repeat download over existing complete model

3. Missing ONNX file triggers repair with custom voices present

4. Download network error and disk-full/write/rename failure

5. SUPERTONIC_CACHE_DIR override and explicit model_dir

6. SUPERTONIC_CUSTOM_STYLES_DIR inside/outside model root

7. Existing stage/backup directories and retry after an interrupted operation

#### Risks

1. Directory rename semantics differ on Windows and when sessions hold files open.

2. Migration/copy logic must not replace user voices with model snapshot files on name collision.

3. Changes in storage paths can strand existing voices unless migration and fallback discovery are implemented.

#### Rollback

Keep the previous directory and user-style backup until acceptance succeeds. A code rollback cannot recover data already deleted by the original implementation; restore those files from external originals/backups. Do not restore a version with unconditional rmtree while allowing model redownload over user data.

#### Repair validation status

NOT RUN — proposed changes and regression tests do not exist yet.

### SPY-2 Medium: Batch voice defaults silently select M1 instead

**Status:** OPEN. [Pinned source](https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/server/schemas.py#L55).

#### Evidence classification

Confirmed through the validation described below

#### Observed mechanism

BatchItem.voice defaults to nonempty M1. routes.py:365 chooses item.voice before defaults.voice, so an omitted item voice never inherits defaults.voice.

#### Impact and scope

A valid batch request with defaults.voice=F1 and items containing only text returns audio from M1, silently using the wrong requested voice.

#### Repair direction

Make BatchItem.voice optional with a None default, then resolve item voice -> defaults.voice -> M1 in the route. Add a test for omitted item voice plus batch default and an explicit item override.

#### Verification already obtained

Confirmed through the real FastAPI app/TestClient with FakeTTS: POST /v1/tts/batch using defaults.voice=F1 returned HTTP 200, but the synthesizer received M1. Reproducer saved.

#### Locations

##### Item 1

###### File

supertonic/server/schemas.py

###### Lines

1. 55

2. 57

###### Url

<https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/server/schemas.py#L55>

#### Change surface

1. supertonic/server/schemas.py: BatchItem.voice

2. supertonic/server/routes.py: synth_batch precedence

3. tests/test_server_routes.py: batch route tests

4. Generated OpenAPI schema expectations if checked

#### Repair steps

1. Represent an omitted item voice as None rather than eagerly filling M1 in BatchItem.

2. Keep resolution order explicit: item override, then defaults.voice, then M1. Decide whether explicit null means inheritance and document it consistently.

3. Exercise the actual route with FakeTTS recording the resolved style; cover omission and overrides together.

#### Invariants

1. A client that omits all voice settings continues to receive M1.

2. An explicit item voice takes precedence over the batch default.

3. Unknown resolved voices continue to return the existing structured 400 error.

#### Acceptance criteria

1. items=[{text: hello}], defaults.voice=F1 invokes F1.

2. An item with voice=M1 overrides defaults.voice=F1.

3. Omitting both defaults.voice and item.voice invokes M1.

4. An unknown inherited default fails clearly without silently switching to M1.

#### Regression scenarios

1. Single and multiple batch items

2. Mixed explicit and omitted item voice

3. Missing defaults object

4. Explicit null/empty voice according to chosen validation policy

5. Unknown default and explicit unknown item voice

#### Risks

1. Changing the item field to optional affects OpenAPI nullability and any schema-generated clients. Preserve request compatibility and the effective M1 fallback.

#### Rollback

Revert only the schema/route change if compatibility tests fail, and temporarily require explicit voice per item. No stored-data migration is involved.

#### Repair validation status

NOT RUN — current bug confirmed with fake TTS; fixed behavior has not been implemented or tested.

### SPY-3 Medium: Chunked JSON imports bypass the 1 MiB request limit

**Status:** OPEN. [Pinned source](https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/server/routes.py#L232).

#### Evidence classification

Confirmed through the validation described below

#### Observed mechanism

StyleImportSizeLimit passes requests without Content-Length (app.py:65-67). The JSON branch then calls request.json() without a bounded read; unlike the file branch, it never checks MAX_STYLE_IMPORT_BYTES. FastAPI multipart dependencies may also buffer/spool before handler file.read, so enforcing the actual incoming ASGI byte count is preferable.

#### Impact and scope

A reachable HTTP client can submit arbitrarily large JSON uploads to a server that claims a 1 MiB cap, causing unbounded body buffering and parsing. The default server is loopback-only; risk grows when configured for a LAN/service binding.

#### Repair direction

Enforce a cumulative receive-byte limit in the ASGI middleware for all import content types, independent of Content-Length, and reject oversized bodies before parsing.

#### Verification already obtained

Confirmed via TestClient content iterator (Transfer-Encoding chunked): 1,048,704-byte JSON with a valid small style plus an ignored large field was accepted with HTTP 200. No large resource-exhaustion payload was used.

#### Locations

##### Item 1

###### File

supertonic/server/routes.py

###### Lines

1. 232

2. 242

###### Url

<https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/server/routes.py#L232>

#### Change surface

1. supertonic/server/app.py: StyleImportSizeLimit

2. supertonic/server/routes.py: import_style request parsing

3. tests/test_server_routes.py: import byte-limit regressions

#### Repair steps

1. Retain the Content-Length preflight for early rejection, but count actual body bytes as the import request is received.

2. At a 1 MiB limit, a simple bounded pre-read with replay to the downstream app can be easier to reason about than letting FastAPI consume an unbounded stream; reject at limit+1 before JSON/multipart parsing.

3. Alternatively wrap ASGI receive with a counting boundary and a dedicated overflow control path; ensure the response is sent once and no success response can be started after overflow.

4. Enforce the same byte definition for JSON and multipart including the policy on multipart envelope overhead. Handle split messages and client disconnect without retaining files or partial styles.

5. Return the existing OpenAI-shaped 413 payload_too_large envelope; leave normal route behavior unchanged.

#### Invariants

1. No import path buffers/parses more than the stated cumulative body limit before rejection.

2. Content-Length absence, understated length, different chunk sizes and content type do not bypass the limit.

3. Rejected imports do not create or overwrite a custom style file.

4. Other routes and non-HTTP ASGI events remain unaffected by this narrowly scoped limit.

#### Acceptance criteria

1. The recorded 1,048,704-byte JSON iterator request returns 413 rather than 200.

2. Under-limit known-length and chunked JSON/multipart imports still work.

3. An exactly-at-limit request follows a documented inclusive rule; limit+1 rejects.

4. An over-limit body does not reach the style save operation or leave disk-spooled upload residue owned by the application.

#### Regression scenarios

1. Content-Length missing/invalid/over-limit

2. Body arrives as one chunk and many small chunks

3. Multipart envelope overhead and a file straddling limit

4. JSON with large ignored fields

5. Disconnect mid-request

6. Rejected overwrite leaves existing custom style intact

#### Risks

1. ASGI middleware ordering and receive replay can break multipart parsing or duplicate responses if not tested.

2. A body-counting fix is a specific request boundary; it does not add authentication, rate limits or global resource controls.

3. Claims about multipart parser buffering were source-informed; only the JSON bypass was dynamically demonstrated.

#### Rollback

Revert middleware changes if valid clients break, retain loopback-only exposure and enforce an actual byte cap in an upstream proxy until a corrected implementation is accepted. A limit-only rollback needs no data migration.

#### Repair validation status

NOT RUN — the original cap bypass is reproduced; no fixed middleware has been built.

### SPY-4 Medium: Long individual sentences bypass max_chunk_length

**Status:** OPEN. [Pinned source](https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/utils.py#L206).

#### Evidence classification

Confirmed through the validation described below

#### Observed mechanism

When a sentence exceeds max_len, the function assigns it unchanged to current_chunk. There is no fallback split at words/characters; sentence splitting only recognizes ASCII .!? followed by whitespace.

#### Impact and scope

The documented maximum chunk length is not enforced for pasted unpunctuated text or Japanese text with normal sentence punctuation. These inputs are sent as one large model inference despite the selected limit. Exact memory/latency impact was not measured.

#### Repair direction

After sentence splitting, split overlong sentences at word boundaries, with a bounded character fallback for text without spaces, including Unicode punctuation handling.

#### Verification already obtained

Confirmed real chunk_text('word '*400, max_len=300) returns one chunk of 1,999 characters. Same bug exists in supertonic/py/helper.py:418-427.

#### Locations

##### Item 1

###### File

supertonic/utils.py

###### Lines

1. 206

2. 215

###### Url

<https://github.com/wwongw21/supertonic-py/blob/908a56486e821e833a80530ff0cae3ad0b046fce/supertonic/utils.py#L206>

#### Change surface

1. supertonic/utils.py: chunk_text

2. supertonic/pipeline.py: chunk_text caller

3. tests/test_core.py and tests/test_utils.py

#### Repair steps

1. Define maximum length over Python Unicode characters and preserve the existing paragraph/sentence ordering and abbreviation rules.

2. For a sentence longer than max_len, split at available whitespace boundaries; if an individual span still exceeds the limit, split it by bounded character positions.

3. Recognize common Unicode sentence punctuation so Japanese text can retain useful boundaries; hard fallback must still enforce the maximum when punctuation/whitespace is absent.

4. Return nonempty bounded chunks without dropping, reordering or duplicating input text. Preserve the documented whitespace-normalization behavior.

5. Add regressions for unpunctuated words, one long word, Japanese text without spaces, Korean text, abbreviations and exact limits. Coordinate behavior with the sibling Python implementation without introducing a cross-repository runtime dependency.

#### Invariants

1. Every returned chunk is nonempty and has len(chunk) <= configured max_len.

2. Original meaningful characters retain order and are emitted exactly once.

3. Empty/whitespace input and existing sentence/paragraph behavior remain defined.

4. Minimum-limit validation remains consistent with the public SDK or example contract.

#### Acceptance criteria

1. chunk_text('word '*400,max_len=300) returns multiple chunks, all <=300, instead of [1999].

2. A Japanese passage longer than 120 characters with no ASCII whitespace yields bounded chunks while preserving all characters.

3. One token longer than the limit is split safely and recombines to the original token.

4. Existing abbreviation and paragraph tests remain green.

#### Regression scenarios

1. Input length max_len-1/max_len/max_len+1

2. One very long word and one long sentence

3. Japanese 。！？ punctuation without following spaces

4. Korean decomposed/combined characters

5. Emoji, combining marks and expression-tag text

6. Empty input, repeated whitespace, paragraphs and common abbreviations

#### Risks

1. A strict character fallback may split a grapheme cluster or expression tag, affecting synthesis quality. Prefer safe text boundaries while maintaining a real length bound; define behavior for indivisible tokens explicitly.

2. More chunks add pauses and inference overhead. Exact memory/latency improvements and audio quality have not been measured.

#### Rollback

Revert the chunker change if it drops text or materially degrades speech; clients can pre-split bounded passages temporarily. No data format migration is required.

#### Repair validation status

NOT RUN — only the current oversized result was observed; no repair or fixed regression suite has run.

## Review date

2026-10-22

## Owner

wwongw21

## Handoff status

All findings OPEN; runtime repairs and acceptance checks are still PROPOSED. Only the portable diagnostic and handoff evidence were added.

## Evidence boundary

Source at the pinned commit, recorded lightweight tests and focused bug reproductions. Fake inference does not establish real synthesis quality or native runtime compatibility.

## Authorization boundary

This handoff describes proposed changes only. Source repositories and remote services were not modified.

## Evidence artifacts

1. /workspace/review/supertonic.json

2. /workspace/review/repro_supertonic.py

## Environment

### Execution

Linux managed cloud environment

### Recorded dependency versions

#### Python

3.12.14

#### Numpy

2.3.5

#### Pytest

9.1.1

#### Onnxruntime

1.31.0

#### Soundfile

0.14.0

#### Huggingface hub

2.2.0

#### Fastapi

0.141.1

#### Pydantic

2.13.4

#### Python multipart

0.0.32

#### Httpx

0.28.1

### Versions note

Versions were read from the original verification venv, which used --system-site-packages and was therefore not isolated from preinstalled host packages. NumPy, FastAPI, Pydantic and httpx were inherited preinstalled packages, not installed by the recorded pip command. No project dependency manifest or lock file changed.

## Setup

### Working directory

/workspace

### Commands

1. ```
   python -m venv --system-site-packages /workspace/review/supertonic-venv
   ```

2. ```
   /workspace/review/supertonic-venv/bin/pip install pytest onnxruntime soundfile huggingface-hub python-multipart -q
   ```

### Status

RAN during original review; do not rerun to claim additional coverage.

### Model requirement

The focused reproducer needs both local repositories at their pinned commits but downloads no model. The offline SDK unit selection also downloads no model.

### Dependency origins

#### Inherited preinstalled packages

1. numpy 2.3.5

2. fastapi 0.141.1

3. pydantic 2.13.4

4. httpx 0.28.1

#### Explicit install command requested

1. ```
   pytest
   ```

2. ```
   onnxruntime
   ```

3. ```
   soundfile
   ```

4. ```
   huggingface-hub
   ```

5. ```
   python-multipart
   ```

#### Qualification

The historical pip command did not install a complete clean-host test environment; --system-site-packages made the listed preinstalled packages available. The explicit request list is not a claim that every transitive dependency was freshly installed.

## Branch

main

## Source directory

/workspace/repos/supertonic-py

## Desired outcome

Preserve imported user voices across model repair/download, honor batch voice selection, enforce the actual style-import byte limit, and keep every inference text chunk within its documented maximum.

## Coverage

Python SDK engine, text utilities, model/voice loader, CLI, FastAPI routes/schemas/style storage, dependency metadata, test and publish workflow configuration. No AGENTS.md present.

## Current mitigations and limits

### Item 1

#### Mitigation

Back up custom voice JSONs to a directory outside every model/cache root before running supertonic download or repairing missing assets. Set SUPERTONIC_CUSTOM_STYLES_DIR to an external persistent location for subsequent imports.

#### Addresses

1. SPY-1

#### Limit

Does not recover already deleted files; originals/backups may be needed. An override still inside the replaced model directory is ineffective.

### Item 2

#### Mitigation

Send the desired voice explicitly on every batch item.

#### Addresses

1. SPY-2

#### Limit

Client workaround only; defaults.voice remains broken until repaired.

### Item 3

#### Mitigation

Keep the default loopback binding. If a service must be exposed, use a proxy that enforces actual received-body size including chunked bodies.

#### Addresses

1. SPY-3

#### Limit

Loopback reduces who can reach the service; it does not implement the missing application limit. Content-Length-only proxy checks are insufficient.

### Item 4

#### Mitigation

Pre-split long text into bounded strings at the client and synthesize them separately.

#### Addresses

1. SPY-4

#### Limit

Clients must preserve text order and intended pauses. No measured performance limit was established.

## Tests and reproductions

### Item 1

#### Name

SDK/core/server offline selection

#### Working directory

/workspace/repos/supertonic-py

#### Command

```
HF_HUB_OFFLINE=1 /workspace/review/supertonic-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_core.py tests/test_utils.py tests/test_validation.py tests/test_config.py tests/test_imports.py tests/test_server_routes.py tests/test_cli_serve.py
```

#### Prerequisites

Isolated venv above; PYTHONPATH resolved by running at repository root; optional server dependencies and httpx installed.

#### Actual result

PASS — 86 tests passed in 0.97 seconds during original review.

#### Proves

Selected utility, mask/style/text validation, configuration/import, HTTP route/encoding and serve parser behavior using fake TTS.

#### Does not prove

Model inference, GPU/native compatibility, full integration suite or remediation of open findings.

### Item 2

#### Name

Combined bug demonstrator, SDK sections

#### Working directory

/workspace

#### Command

```
HF_HUB_OFFLINE=1 /workspace/review/supertonic-venv/bin/python /workspace/review/repro_supertonic.py
```

#### Prerequisites

Both supertonic-py and supertonic repositories at pinned paths; same isolated venv.

#### Actual result

EXIT 0. SDK output: batch defaults F1: 200 actual voice: M1; chunked JSON bytes: 1048704 status: 200; custom voice after successful download exists: False; SDK chunk length [1999] for max_len=300.

#### Proves

Three asserted SDK bugs plus one observed bounded-chunking contract failure. The script also asserts one separate supertonic examples bug; do not count it as an SDK test.

#### After repair

These assertions intentionally encode vulnerable/current behavior. They must fail or be replaced with repair assertions after fixing the code; a green original bug demonstrator is not a release gate.

### Item 3

#### Name

Full-suite attempt

#### Working directory

/workspace/repos/supertonic-py

#### Command

```
/workspace/review/supertonic-venv/bin/python -m pytest -q -m 'not slow and not integration' -p no:cacheprovider
```

#### Actual result

INCOMPLETE — interrupted with KeyboardInterrupt; 2 passed in 13.21 s before interruption. Unmarked model-dependent tests began downloading model assets.

#### Limit

Do not report a completed full-suite pass or infer markers exclude model tests.

## Remaining acceptance not run

### Item 1

#### Status

NOT RUN

#### Scenario

With temporary real model cache and a valid imported custom voice, run real CLI download and missing-model auto-repair after the repair; verify voice bytes, listability and synthesis survive.

#### Prerequisites

Model network/download access, storage for assets, safe disposable cache and fixed code.

### Item 2

#### Status

NOT RUN

#### Scenario

Real uvicorn service with native and OpenAI-compatible requests; verify batch voice output, under/over-limit JSON and multipart imports using actual chunked HTTP transfer.

#### Prerequisites

[serve] installation, model assets, loopback service and bounded local HTTP client.

### Item 3

#### Status

NOT RUN

#### Scenario

Real long English, Japanese and Korean synthesis with bounded chunks; compare final text coverage and pauses.

#### Prerequisites

Selected model and voice assets, CPU runtime and human playback inspection.

### Item 4

#### Status

NOT RUN

#### Scenario

Supported Python/OS matrix, wheel/sdist installation and optional-extra smoke checks.

#### Prerequisites

Python 3.9–3.13 runners on Linux/macOS/Windows and appropriate native wheels.

## Operational notes

1. The README contains an upstream archive/support notice. Confirm that this fork owns maintenance and releases; do not assume upstream will publish these repairs.

2. Root recorded no GitHub Actions runs on these forks. The workflow files define intended checks, but their presence does not establish a passing run.

3. No GPU/native/browser/mobile or packaged release checks ran.

## All findings open

true

## Implementation performed

false

## Proposed clean host setup

### Status

PROPOSED — NOT RUN or validated; supplied as clean-host installation guidance only.

### Source of constraints

pyproject.toml at 908a56486e821e833a80530ff0cae3ad0b046fce: existing [project.optional-dependencies].serve plus pytest/httpx floors from dev; no new extra is proposed.

### Working directory

/workspace/repos/supertonic-py at the pinned reviewed commit

### Prerequisites

1. Python >=3.9 supported by project metadata; Python 3.12 is the closest match to the recorded run.

2. pip/venv support and network access to resolve declared packages; native wheels compatible with the host platform.

3. Both reviewed repository checkouts and the saved reproducer if the combined bug probe will be run.

### Commands

1. ```
   python -m venv /workspace/review/supertonic-clean-venv
   ```

2. ```
   /workspace/review/supertonic-clean-venv/bin/python -m pip install -e '.[serve]' 'pytest>=7.0' 'httpx>=0.27'
   ```

### Dependency constraints from project

#### Base

1. onnxruntime>=1.19.0; python_version < '3.13'

2. onnxruntime>=1.20.0; python_version >= '3.13'

3. numpy>=1.21.0

4. soundfile>=0.10.0

5. huggingface-hub>=0.10.0

#### Serve

1. fastapi>=0.110

2. uvicorn[standard]>=0.27

3. pydantic>=2.5

4. python-multipart>=0.0.9

#### Test additions

1. pytest>=7.0

2. httpx>=0.27

### Expected scope

Install package source plus the declared optional server dependencies and lightweight test dependencies into a venv without inherited system-site-packages. No model download or test is part of these setup commands.

### Result

NOT RUN. A fresh dependency resolver may choose versions different from the original recorded environment. The original 86-test pass must not be attributed to this proposed clean-host setup.

## Delivery and diagnostic files

### Destination repository

wwongw21/supertonic-py

### Status

All findings remain OPEN; documentation transfer only.

### Portable evidence

The historical `/workspace` commands and outputs remain unchanged as records of the original review. The new portable run is separate; it records its UTC time, Python/OS, source pin, and LF-normalized SHA-256 hashes for every source file it executes.

### Diagnostics included

Run from the repository root with Python 3.9 or later:

```text
python docs/reviews/2026-10-22/diagnostics/repro_pinned_sources.py
```

The probe uses only the Python standard library. It AST-extracts the pinned downloader, batch voice default and resolver, import middleware and handler, and chunking helper. It mocks the Hugging Face download, style store, and request/ASGI boundary. No HTTP client, model package, model weights, network request, or application file write is used. The source hash checks normalize CRLF to LF, so they work from Windows and LF checkouts.

The captured run reproduced these pre-fix results: the downloader removed a seeded custom voice while publishing a mocked snapshot; the batch resolver selected M1 when the batch default was F1 and the item omitted its voice; a 1,048,704-byte chunked JSON body without Content-Length reached the mocked handler and returned its success status; and `chunk_text("word " * 400, max_len=300)` returned a 1,999-character chunk. The SPY-3 status comes from the mocked handler boundary, not an HTTP client run.

The evidence is [portable-probe-2026-10-22.txt](evidence/portable-probe-2026-10-22.txt). It is a new source-level reproduction and does not replace the historical run results above.

### Not included

Original environments, model weights, user data and complete source snapshots are not included. Clone the pinned source and follow the recorded prerequisites. Bug-demonstration scripts intentionally assert pre-fix behavior; they are not a post-fix passing suite.

### Combined probe

The historical `repro_supertonic.py` still is not included. This repository now contains a self-contained probe for its four recorded `supertonic-py` observations; the sibling `supertonic` repository has a separate probe for the two Python example observations. Each run checks its own pinned source hashes.


## Remaining review limits

1. No real model inference, synthesis quality evaluation, GPU execution, browser UI, mobile app, full native-language builds or packaged release testing.

2. Tests use deterministic fake inference outputs and temporary user-style directories where appropriate; these reproduce control-flow/contracts rather than audio quality.

3. No GitHub Actions runs exist on these forks according to root's connector query; this is absent evidence, not a failing CI result.

4. Runtime source files were not edited; the historical test imports may generate ignored __pycache__ files.

