# Medical PSI Prototype

A Private Set Intersection (PSI) protocol for medical image retrieval built on Microsoft APSI via the PyAPSI bindings. The client can query the server's medical database using an X-ray image without revealing the image, while the server keeps its database private.

## Architecture

- Core Logic: Microsoft APSI (PyAPSI bindings)
- Image Processing: ImageTokenizer (CNN-based hashing placeholder)
- Server: Flask (REST API)
- Protocol: OPRF-based sender-augmented PSI

## Setup

### 1. Docker (Recommended)

```bash
docker build -t psi-medical Code/psi-medical-project
# Run server
cd Code/psi-medical-project
docker run -p 5000:5000 -v $(pwd)/server/data:/app/server/data psi-medical
# Run client
docker run -it --network="host" psi-medical python -m client.client_app
```

### 2. Local Development

1. Clone `LGro/PyAPSI` and install dependencies (`cmake`, `g++`).
2. Run `pip install .` inside the PyAPSI repo.
3. Install project requirements with `pip install -r requirements.txt`.

## Usage

1. Server indexes dummy patient data on startup.
2. Client points to an image path and runs the PSI protocol.
3. Matches trigger metadata fetches from `/metadata/<record_id>`.

## Directory Overview

- `client/`: Image processing utilities and the PSI client.
- `server/`: APSI database manager and Flask server.
- `common/`: Shared configuration and helper functions.
- `tests/`: Integration test bypassing HTTP to validate protocol flow.
