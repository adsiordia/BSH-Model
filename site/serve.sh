#!/usr/bin/env bash
# Serve the explorer on this machine. Reach it from your laptop with an SSH tunnel:
#     ssh -N -L 8811:localhost:8811 adsiordia@gpu.jinich.ucsd.edu
# then open http://localhost:8811
#
#   ./serve.sh          start (or report that it is already running)
#   ./serve.sh stop     stop it
#   ./serve.sh status   check
set -euo pipefail
PORT="${PORT:-8811}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$DIR/.serve.pid"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-start}" in
  stop)
    if running; then kill "$(cat "$PIDFILE")"; rm -f "$PIDFILE"; echo "stopped"
    else echo "not running"; fi ;;
  status)
    if running; then echo "running, pid $(cat "$PIDFILE"), port $PORT"
    else echo "not running"; fi ;;
  start)
    if running; then echo "already running, pid $(cat "$PIDFILE"), port $PORT"; exit 0; fi
    [[ -f "$DIR/index.html" ]] || { echo "index.html missing — run: python build.py" >&2; exit 1; }
    # bind to loopback only: reachable through the SSH tunnel, not from the network
    ( setsid nohup python -m http.server "$PORT" --bind 127.0.0.1 --directory "$DIR" \
        > "$DIR/.serve.log" 2>&1 < /dev/null & echo $! > "$PIDFILE" )
    sleep 1
    if running; then
      echo "serving $DIR on 127.0.0.1:$PORT  (pid $(cat "$PIDFILE"))"
      echo
      echo "from your laptop:"
      echo "  ssh -N -L $PORT:localhost:$PORT $USER@$(hostname)"
      echo "  then open http://localhost:$PORT"
    else echo "failed to start — see $DIR/.serve.log" >&2; exit 1; fi ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 1 ;;
esac
