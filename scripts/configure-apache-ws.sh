#!/bin/bash
# Configures Apache to proxy WebSocket connections to FastAPI (uvicorn).
# Run as root on the server.
set -e

echo "=== Enabling Apache modules ==="
a2enmod proxy_wstunnel rewrite
echo ""

# Find the site config that contains the API ProxyPass
echo "=== Locating site config ==="
CONFIG=""
for f in /etc/apache2/sites-enabled/*.conf /etc/apache2/sites-enabled/*; do
    [ -f "$f" ] && grep -q "ProxyPass /api" "$f" 2>/dev/null && CONFIG="$f" && break
done
if [ -z "$CONFIG" ]; then
    for f in /etc/apache2/sites-enabled/*.conf /etc/apache2/sites-enabled/*; do
        [ -f "$f" ] && grep -q "miner.noxiousgaming\|8000" "$f" 2>/dev/null && CONFIG="$f" && break
    done
fi
if [ -z "$CONFIG" ]; then
    echo "ERROR: Could not auto-detect site config."
    echo "Please add these lines manually BEFORE your 'ProxyPass /api/' line:"
    echo ""
    echo "    RewriteEngine On"
    echo "    RewriteCond %{HTTP:Upgrade} websocket [NC]"
    echo "    RewriteCond %{HTTP:Connection} Upgrade [NC]"
    echo "    RewriteRule ^/api/(.*) ws://127.0.0.1:8000/api/\$1 [P,L]"
    exit 1
fi
echo "Found: $CONFIG"
echo ""

# Check if WebSocket rules already present
if grep -qE "proxy_wstunnel|RewriteCond.*[Uu]pgrade.*websocket|upgrade=websocket" "$CONFIG"; then
    echo "WebSocket rules already present in config."
else
    echo "=== Adding WebSocket proxy rules ==="
    cp "$CONFIG" "${CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
    echo "Backup saved."

    # Insert the WebSocket RewriteRules before the first "ProxyPass /api/" line
    RULES='    # WebSocket proxy — must come before HTTP ProxyPass\n    RewriteEngine On\n    RewriteCond %{HTTP:Upgrade} websocket [NC]\n    RewriteCond %{HTTP:Connection} Upgrade [NC]\n    RewriteRule ^\/api\/(.*) ws:\/\/127.0.0.1:8000\/api\/$1 [P,L]\n'
    sed -i "0,/ProxyPass \/api\//{ s|ProxyPass /api/|${RULES}\n    ProxyPass /api/| }" "$CONFIG"
    echo "Rules added."
fi

echo ""
echo "=== Testing config ==="
apache2ctl configtest

echo ""
echo "=== Reloading Apache ==="
systemctl reload apache2
echo ""
echo "Done! WebSocket proxying is now configured."
