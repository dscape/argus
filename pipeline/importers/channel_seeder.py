"""Seed crawl_channels table from channels.yaml config."""

import os

import yaml

from pipeline.db.connection import get_conn

CHANNELS_CONFIG = os.path.join("configs", "pipeline", "channels.yaml")


def seed_channels(config_path: str = CHANNELS_CONFIG):
    """Load channel config from YAML and upsert into crawl_channels."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    channels = config.get("channels", [])
    print(f"Seeding {len(channels)} channels from {config_path}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            for ch in channels:
                channel_id = ch.get("channel_id")
                handle = ch.get("handle")
                name = ch.get("name", handle or "Unknown")
                tier = ch.get("tier", 3)
                notes = ch.get("notes")

                # If no channel_id provided, use handle as placeholder
                # (will be resolved later by channel_resolver)
                if not channel_id:
                    channel_id = f"UNRESOLVED:{handle or name}"

                # Compute uploads playlist ID from channel ID
                uploads_playlist_id = None
                if channel_id.startswith("UC"):
                    uploads_playlist_id = "UU" + channel_id[2:]

                cur.execute(
                    """
                    INSERT INTO crawl_channels
                        (channel_id, channel_handle, channel_name, tier,
                         uploads_playlist_id, enabled, notes)
                    VALUES (%s, %s, %s, %s, %s, true, %s)
                    ON CONFLICT (channel_id) DO UPDATE SET
                        channel_handle = EXCLUDED.channel_handle,
                        channel_name = EXCLUDED.channel_name,
                        tier = EXCLUDED.tier,
                        uploads_playlist_id = COALESCE(
                            crawl_channels.uploads_playlist_id,
                            EXCLUDED.uploads_playlist_id
                        ),
                        notes = EXCLUDED.notes
                    """,
                    (channel_id, handle, name, tier, uploads_playlist_id, notes),
                )

            conn.commit()
            print(f"Seeded {len(channels)} channels")


if __name__ == "__main__":
    seed_channels()
