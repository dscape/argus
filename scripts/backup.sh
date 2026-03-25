#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ───────────────────────────────────────────
BACKUP_ROOT="$HOME/argus-backups"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RETENTION_DAYS=7
TODAY=$(date +%Y-%m-%d)
BACKUP_DIR="$BACKUP_ROOT/$TODAY"

# ── Helpers ─────────────────────────────────────────────────
info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; exit 1; }

# ── Pre-checks ──────────────────────────────────────────────
if ! docker compose -f "$PROJECT_DIR/docker-compose.yaml" ps postgres --format '{{.Status}}' 2>/dev/null | grep -qi "up"; then
    error "PostgreSQL container is not running. Start it with 'make db-up' first."
fi

if [ ! -d "$PROJECT_DIR/data" ]; then
    error "data/ directory not found at $PROJECT_DIR/data"
fi

# ── Create backup directory ─────────────────────────────────
mkdir -p "$BACKUP_DIR"

# ── 1. PostgreSQL dump ──────────────────────────────────────
info "Backing up PostgreSQL database..."
docker compose -f "$PROJECT_DIR/docker-compose.yaml" exec -T postgres \
    pg_dump -U argus argus | gzip > "$BACKUP_DIR/argus_db.sql.gz"
info "Database backup: $(ls -lh "$BACKUP_DIR/argus_db.sql.gz" | awk '{print $5}')"

# ── 2. Data folder archive ──────────────────────────────────
info "Backing up data/ directory..."
tar -czf "$BACKUP_DIR/argus_data.tar.gz" -C "$PROJECT_DIR" data/
info "Data backup: $(ls -lh "$BACKUP_DIR/argus_data.tar.gz" | awk '{print $5}')"

# ── 3. Record timestamp ────────────────────────────────────
date +%s > "$BACKUP_ROOT/.last_backup"

# ── 4. Prune old backups ───────────────────────────────────
info "Pruning backups older than $RETENTION_DAYS days..."
find "$BACKUP_ROOT" -maxdepth 1 -type d -name "20*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true

# ── Done ────────────────────────────────────────────────────
echo ""
echo "Backup complete: $BACKUP_DIR"
ls -lh "$BACKUP_DIR"
echo ""
echo "To restore:"
echo "  DB:   gunzip < $BACKUP_DIR/argus_db.sql.gz | docker compose exec -T postgres psql -U argus argus"
echo "  Data: tar -xzf $BACKUP_DIR/argus_data.tar.gz -C $PROJECT_DIR"
