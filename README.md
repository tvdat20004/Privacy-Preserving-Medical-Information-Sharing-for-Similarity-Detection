# Medical PSI Prototype

A Private Set Intersection (PSI) protocol for medical image retrieval built on Microsoft APSI via the PyAPSI bindings. The client can query the server's medical database using an X-ray image without revealing the image, while the server keeps its database private.

## Architecture

- Core Logic: Microsoft APSI (PyAPSI bindings)
- Image Processing: ImageTokenizer (CNN-based hashing placeholder)
- Server: Flask (REST API)
- Protocol: OPRF-based sender-augmented PSI

## Features

- Private Set Intersection (PSI) for medical X-ray queries using Microsoft APSI (PyAPSI bindings)
- Client flow: offline encode image → download `.enc` tokens or forward to search; upload `.enc`/tokens to run PSI and view top matches with metadata
- Admin flow: login (demo `admin/12345`), view metadata, upload new record (image + diagnosis/age/name) to tokenize and index, delete records with confirmation (rebuilds DB from stored tokens)
- Data handling: server keeps DB/tokens private; client image is tokenized locally/server-side without exposing raw data externally

## How it works (summary)

- Tokenization: MedicalImageTokenizer (torch + torchvision) produces fixed-length tokens from an X-ray
- PSI protocol: LabeledClient/LabeledServer (PyAPSI) run OPRF + query/extract locally; matches are counts per record ID
- Metadata join: top matches are enriched with metadata from `server/data/metadata.json`
- Persistence: tokens/metadata/DB stored under `server/data/` (`medical.db`, `metadata.json`, `tokens.json`)

## Directory Overview

- `client/`: Image processing utilities and the PSI client.
- `server/`: APSI database manager and Flask server.
- `common/`: Shared configuration and helper functions.
- `tests/`: Integration test bypassing HTTP to validate protocol flow.
