# Twitch Miner Backend

FastAPI-Backend als Multi-User-Orchestrator für [TwitchDropsMiner](https://github.com/rangermix/TwitchDropsMiner) und [TwitchChannelPointsMiner V2](https://github.com/rdavidoff/Twitch-Channel-Points-Miner-v2). Beide Miner laufen als Docker-Container, verwaltet pro User über eine REST-API mit JWT-Authentifizierung.

**Frontend:** [twitch-miner-frontend](https://github.com/CptObvs/twitch-miner-frontend)

## Miner-Typen

| Typ | Beschreibung |
|-----|-------------|
| **TwitchDropsMiner** | Drop-Farming mit Web-UI, erreichbar via Backend-Proxy |
| **TwitchChannelPointsMiner V2** | Channel-Points-Farming, Logs persistent in `output.log` |

Docker-Images werden über `.env` konfiguriert (`DOCKER_IMAGE`, `V2_DOCKER_IMAGE`).

## Drops-Miner aktualisieren

`DOCKER_IMAGE=rangermix/twitch-drops-miner:latest` bleibt der Standard. Vor jedem neuen Drops-Container-Start führt die API `docker pull` aus. Erst nach einem erfolgreichen Pull wird der vorherige Container ersetzt; ein fehlgeschlagenes Update wird als Startfehler gemeldet. Für lokale Images oder Offline-Betrieb lässt sich das mit `DOCKER_PULL_ON_START=false` abschalten. `DOCKER_PULL_TIMEOUT` begrenzt die Wartezeit, standardmäßig auf 300 Sekunden.

Der Log-Eintrag `Version: 1.2.5` zeigt einen alten Miner mit Gerätecode-Login. Seit Version 2.x erfolgt die Twitch-Anmeldung über die Browseransicht im Dashboard. Ein Update eines anderen Containers oder ein Neustart allein aktualisiert die von dieser API verwalteten Nutzercontainer nicht. Die offiziellen [Release-Hinweise](https://github.com/rangermix/TwitchDropsMiner/releases/tag/v2.2.2) beschreiben den neuen Login.

Auf dem Server den geänderten API-Code übernehmen, in der tatsächlich geladenen `.env` `DOCKER_IMAGE=rangermix/twitch-drops-miner:latest` setzen und die API neu starten. Die Migration läuft beim API-Start automatisch, sofern `RUN_MIGRATIONS_ON_STARTUP=true` ist. Anschließend bestehende Drops-Instanzen über die API stoppen und wieder starten, ihre Extra-UI neu laden und bei Bedarf dort Twitch anmelden. Der vorhandene `/app/data`-Mount bleibt erhalten; Cookies und Einstellungen werden nicht gelöscht.

Falls der Server über `scripts/start-prod.sh` startet: Dieses Script setzt den Checkout auf den Stand von `origin` zurück. Die Änderungen müssen deshalb vor diesem Start committed und zum verwendeten Remote-Branch gepusht sein.

### HTTPS-Reverse-Proxy und Fehler 403 beim Login

Wenn die API hinter einem HTTPS-Proxy intern per HTTP angesprochen wird, kann der Browser-Login mit `Invalid dashboard origin` und WebSocket-Code 1006 scheitern. In der tatsächlich geladenen Server-Umgebung die öffentliche Origin setzen und die API neu starten:

```dotenv
UI_PUBLIC_ORIGIN=https://miner.noxiousgaming.eu
```

Das ist nur Schema plus Domain und gegebenenfalls Port, ohne `/api`, Instanz-Pfad oder Token. Die feste Origin wird für HTTP und WebSocket sowie für sichere Cookies verwendet. Andere Origins werden weiterhin abgelehnt. Wenn der Reverse-Proxy `Host` und `X-Forwarded-Proto` korrekt weitergibt, ist die Einstellung optional. Der Proxy muss WebSocket-Upgrades auch für den Instanz-Pfad `/api/session/vnc` durchreichen.

Die neuen Container verwenden `--init`, 256 MiB Shared Memory und die konfigurierte Stop-Wartezeit. `DOCKER_TIMEZONE` ist standardmäßig `Europe/Berlin` und muss zur Zeitzone des Internetanschlusses des Miner-Servers passen. Der Browser-Login nutzt denselben authentifizierten UI-Proxy: HTTP-Origins, Login-Weiterleitungen, noVNC-Module und binäre WebSocket-Daten werden weitergeleitet. Dashboard-Cookies sind auf die jeweilige Instanz beschränkt; Backend-JWTs gelangen nicht zum Miner. Ein bestehender zusätzlicher Dashboard-Passwortschutz funktioniert weiterhin. Der native Desktop-Helper benötigt eine direkt erreichbare Miner-Adresse; er unterstützt den zusätzlichen API-Login und den Instanz-Unterpfad nicht.

## IP-Tracking und automatische Bans

Die Tracking-Liste und der Speicher für noch nicht gebannte 404-Anfragen enthalten höchstens 1.000 IPs. Die Tracking-Liste behält die zuletzt aktiven Adressen; ältere Einträge werden verdrängt. Migration `0006_limit_ip_tracking` bereinigt vorhandene Tracking-Daten auf 1.000 Einträge und ergänzt Indizes. Diese entfernten historischen Einträge werden bei einem Downgrade nicht wiederhergestellt. Aktive Bans behalten ihre Gültigkeit bis zum Ablauf oder einer manuellen Freigabe.

Unbekannte API-Pfade führen standardmäßig nach **5 Aufrufen innerhalb von 60 Sekunden** zu einem **einstündigen Ban**. Eindeutige Scanner-Pfade wie `/.env`, `/.git/config` oder `/wp-login.php` lösen ihn bereits beim ersten 404 aus. 404-Antworten regulärer API-/Miner-Proxy-Routen, `OPTIONS`, Loopback-Adressen und übliche Anfragen nach `favicon.ico`, `robots.txt` oder `sitemap.xml` zählen nicht. Konfiguration:

```dotenv
IP_BAN_HIT_THRESHOLD=5
IP_BAN_WINDOW_SECONDS=60
IP_BAN_DURATION_HOURS=1
```

Tracking-Schreibzugriffe werden alle 30 Sekunden gesammelt als Batch ausgeführt. Fehlgeschlagene Schreibzugriffe werden erneut versucht; beim Herunterfahren werden ausstehende Einträge gespeichert. GeoIP-Abfragen sind auf acht gleichzeitige Aufgaben begrenzt und werden nur für öffentliche Adressen gestartet. Normale 404-Aufrufe benötigen keine Datenbankverbindung; ein Ban wird mit einem einzelnen Upsert gespeichert. Setup, Migration und API verwenden dieselbe `DATABASE_URL`.

## Tests

```bash
python -m pytest -q
```

Die Regressionstests prüfen Image-Updates und deren Fehlerfälle, den Dashboard-/noVNC-Proxy einschließlich Cookie-Trennung und binärer WebSockets sowie IP-Retention, Migrationen und Ban-Regeln. Docker-Aufrufe werden isoliert simuliert; ein produktiver Twitch-Login ist kein Bestandteil der Tests.
