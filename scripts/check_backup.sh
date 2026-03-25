#!/usr/bin/env bash
# Non-blocking staleness check — prints a warning if last backup is >3 days old.

BACKUP_ROOT="$HOME/argus-backups"
STAMP_FILE="$BACKUP_ROOT/.last_backup"
STALE_SECONDS=$((3 * 86400))

if [ ! -f "$STAMP_FILE" ]; then
    echo ""
    echo "  ⚠  No backups found. Run 'make backup' to create one."
    echo ""
    exit 0
fi

last=$(cat "$STAMP_FILE")
now=$(date +%s)
age=$(( now - last ))

if [ "$age" -gt "$STALE_SECONDS" ]; then
    days=$(( age / 86400 ))
    echo ""
    echo "  ⚠  Last backup was ${days} days ago. Run 'make backup' to back up."
    echo ""
fi

exit 0
