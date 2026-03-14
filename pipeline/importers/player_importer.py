"""Import players from players.zip into fide_players + player_aliases."""

import json
import os
import re
import zipfile

from pipeline.db.connection import get_conn

PLAYERS_ZIP = os.path.join("data", "chess", "players.zip")


def _generate_aliases(name: str) -> list[str]:
    """Generate common name variants for fuzzy matching.

    Input format from FIDE: "Surname, Firstname" or "Firstname Surname".
    """
    aliases = set()
    aliases.add(name)
    aliases.add(name.lower())

    # Handle "Surname, Firstname" format
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        surname, firstname = parts[0], parts[1] if len(parts) > 1 else ""
        if firstname:
            # "Firstname Surname"
            aliases.add(f"{firstname} {surname}")
            aliases.add(f"{firstname} {surname}".lower())
            # "F. Surname"
            aliases.add(f"{firstname[0]}. {surname}")
            aliases.add(f"{firstname[0]}. {surname}".lower())
            # Surname only
            aliases.add(surname)
            aliases.add(surname.lower())
            # Firstname only (if unique enough, >3 chars)
            if len(firstname) > 3:
                aliases.add(firstname)
                aliases.add(firstname.lower())
    else:
        # Handle "Firstname Surname" format
        parts = name.split()
        if len(parts) >= 2:
            firstname = parts[0]
            surname = " ".join(parts[1:])
            # "Surname, Firstname"
            aliases.add(f"{surname}, {firstname}")
            aliases.add(f"{surname}, {firstname}".lower())
            # "F. Surname"
            aliases.add(f"{firstname[0]}. {surname}")
            aliases.add(f"{firstname[0]}. {surname}".lower())
            # Surname only
            aliases.add(surname)
            aliases.add(surname.lower())

    return list(aliases)


def import_players(zip_path: str = PLAYERS_ZIP):
    """Import players from players.zip into the database."""
    with zipfile.ZipFile(zip_path) as z:
        json_files = [n for n in z.namelist() if n.endswith(".json") and not n.startswith("__MACOSX")]
        with z.open(json_files[0]) as f:
            players = json.load(f)

    print(f"Loaded {len(players)} players from {zip_path}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            inserted = 0
            aliases_inserted = 0

            for p in players:
                fide_id_raw = p.get("fideId")
                if not fide_id_raw:
                    continue
                # fideId can be string like "1503014"
                fide_id = int(fide_id_raw) if isinstance(fide_id_raw, str) else fide_id_raw

                name = p.get("name", "")
                if not name:
                    continue

                cur.execute(
                    """
                    INSERT INTO fide_players
                        (fide_id, name, federation, title, standard_rating,
                         rapid_rating, blitz_rating, birth_year, slug)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (fide_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        federation = EXCLUDED.federation,
                        title = EXCLUDED.title,
                        standard_rating = EXCLUDED.standard_rating,
                        rapid_rating = EXCLUDED.rapid_rating,
                        blitz_rating = EXCLUDED.blitz_rating,
                        birth_year = EXCLUDED.birth_year,
                        slug = EXCLUDED.slug
                    """,
                    (
                        fide_id,
                        name,
                        p.get("federation"),
                        p.get("title"),
                        p.get("standardRating"),
                        p.get("rapidRating"),
                        p.get("blitzRating"),
                        p.get("birthYear"),
                        p.get("slug"),
                    ),
                )
                inserted += 1

                # Generate and insert aliases
                all_aliases = _generate_aliases(name)

                # Also add explicit aliases from the data
                for slug_alias in p.get("aliases", []):
                    # Convert slug format to name: "carlsen-magnus" -> "carlsen magnus"
                    # Remove trailing FIDE ID from slugs like "carlsen-magnus-1503014"
                    clean = re.sub(r"-\d+$", "", slug_alias)
                    clean = clean.replace("-", " ")
                    all_aliases.append(clean)

                for alias in all_aliases:
                    alias = alias.strip()
                    if not alias or len(alias) < 2:
                        continue
                    cur.execute(
                        """
                        INSERT INTO player_aliases (alias, fide_id, source)
                        VALUES (%s, %s, 'import')
                        ON CONFLICT (alias, fide_id) DO NOTHING
                        """,
                        (alias, fide_id),
                    )
                    aliases_inserted += 1

            conn.commit()
            print(f"Inserted {inserted} players, {aliases_inserted} aliases")


if __name__ == "__main__":
    import_players()
