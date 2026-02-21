# Twitch Miner Backend — Multi-User (Docker)

> Dieser Branch ist eine Docker-basierte Neuimplementierung.
> Als Miner wird [TwitchDropsMiner](https://github.com/rangermix/TwitchDropsMiner) verwendet — pro Instanz ein eigener Docker-Container.
> Multi-User-Unterstützung (Login, Registration, Admin) wird von diesem Backend bereitgestellt.

## Features

- Multi-user mit JWT-Authentifizierung (Login, Registration per Einladungscode, Admin-Panel)
- Instanz-Verwaltung: Docker-Container starten/stoppen pro User
- Authentifizierter Proxy: Miner-Web-UI (`/instances/{id}/ui`) nur für den Besitzer/Admin erreichbar
- Real-time Log-Streaming via SSE (`docker logs -f`)

## Voraussetzungen

- Python 3.12+
- Docker (muss laufen: `docker info`)

## Quick Start

```bash
# 1. Setup
git clone <your-repo-url>
cd twitch-miner-api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: JWT_SECRET setzen

# 3. Admin erstellen & starten
python setup.py --create-admin
python -m alembic upgrade head
uvicorn main:app --reload
```

**API Docs:** http://localhost:8000/docs (via `DOCS_URL` in `.env` anpassbar)

## Production

```bash
cp .env.prod .env
# Edit .env: JWT_SECRET, CORS_ORIGINS, ENABLE_SWAGGER setzen
./scripts/start-prod.sh
```

## Endpoints

| Methode | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/instances/` | Neue Instanz anlegen (nur DB-Eintrag) |
| `GET` | `/instances/` | Alle eigenen Instanzen auflisten |
| `POST` | `/instances/{id}/start` | Docker-Container starten |
| `POST` | `/instances/{id}/stop` | Docker-Container stoppen |
| `GET` | `/instances/{id}/status` | Status prüfen |
| `GET` | `/instances/{id}/logs` | Live-Logs (SSE, `docker logs -f`) |
| `GET/POST` | `/instances/{id}/ui/*` | Proxy zur Miner-Web-UI (Auth erforderlich) |
| `WS` | `/instances/{id}/ui/*?token=<jwt>` | WebSocket-Proxy (Socket.IO) |

## Miner-Container

Jeder Container wird gestartet mit:
```bash
docker run -d \
  --name miner-{instance_id} \
  -p 127.0.0.1:{port}:8080 \
  -v {data_dir}:/app/data \
  rangermix/twitch-drops-miner:latest
```

Die Container sind nur über `127.0.0.1` erreichbar — öffentlicher Zugriff nur über den Backend-Proxy.

## Log-Streaming

```bash
curl -N -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/instances/{id}/logs
```
