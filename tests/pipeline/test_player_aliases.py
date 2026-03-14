"""Tests for player alias generation (pipeline.importers.player_importer._generate_aliases)."""

import pytest
from pipeline.importers.player_importer import _generate_aliases


class TestGenerateAliasesSurnameFirst:
    """Test alias generation for 'Surname, Firstname' format (FIDE standard)."""

    def test_full_name_preserved(self):
        aliases = _generate_aliases("Carlsen, Magnus")
        assert "Carlsen, Magnus" in aliases

    def test_lowercase_variant(self):
        aliases = _generate_aliases("Carlsen, Magnus")
        assert "carlsen, magnus" in aliases

    def test_firstname_surname_order(self):
        aliases = _generate_aliases("Carlsen, Magnus")
        assert "Magnus Carlsen" in aliases
        assert "magnus carlsen" in aliases

    def test_initial_dot_surname(self):
        aliases = _generate_aliases("Carlsen, Magnus")
        assert "M. Carlsen" in aliases
        assert "m. carlsen" in aliases

    def test_surname_only(self):
        aliases = _generate_aliases("Carlsen, Magnus")
        assert "Carlsen" in aliases
        assert "carlsen" in aliases

    def test_firstname_only_if_long(self):
        aliases = _generate_aliases("Carlsen, Magnus")
        assert "Magnus" in aliases  # len("Magnus") > 3

    def test_short_firstname_excluded(self):
        aliases = _generate_aliases("Lu, Bo")
        assert "Bo" not in aliases  # len("Bo") <= 3


class TestGenerateAliasesFirstnameFirst:
    """Test alias generation for 'Firstname Surname' format."""

    def test_reversed_to_surname_first(self):
        aliases = _generate_aliases("Magnus Carlsen")
        assert "Carlsen, Magnus" in aliases

    def test_initial_dot_variant(self):
        aliases = _generate_aliases("Magnus Carlsen")
        assert "M. Carlsen" in aliases

    def test_surname_only(self):
        aliases = _generate_aliases("Magnus Carlsen")
        assert "Carlsen" in aliases

    def test_multi_word_surname(self):
        """Handles 'Firstname Multi Word Surname' correctly."""
        aliases = _generate_aliases("Ding Liren")
        assert "Liren, Ding" in aliases
        assert "D. Liren" in aliases


class TestGenerateAliasesEdgeCases:
    def test_single_word_name(self):
        aliases = _generate_aliases("Madonna")
        assert "Madonna" in aliases
        assert "madonna" in aliases

    def test_empty_string(self):
        aliases = _generate_aliases("")
        assert "" in aliases  # preserved, but filtered by len() check in importer
