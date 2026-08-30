from __future__ import annotations

import pytest

from backend.reguaz.services.speech.azerbaijani_numbers import (
    integer_to_azerbaijani,
    integer_to_ordinal_azerbaijani,
    verbalize_numbers,
)
from backend.reguaz.services.speech.text_normalizer import prepare_speech_text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "sıfır"),
        (7, "yeddi"),
        (10, "on"),
        (19, "on doqquz"),
        (105, "yüz beş"),
        (500, "beş yüz"),
        (1_000, "min"),
        (1_001, "min bir"),
        (1_000_000, "bir milyon"),
        (500_000_000_000, "beş yüz milyard"),
        (-42, "mənfi qırx iki"),
    ],
)
def test_integer_to_azerbaijani(value: int, expected: str) -> None:
    assert integer_to_azerbaijani(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, "birinci"),
        (3, "üçüncü"),
        (6, "altıncı"),
        (20, "iyirminci"),
        (2026, "iki min iyirmi altıncı"),
    ],
)
def test_integer_to_ordinal_azerbaijani(value: int, expected: str) -> None:
    assert integer_to_ordinal_azerbaijani(value) == expected


def test_large_compact_and_hybrid_scale_numbers_are_verbalized() -> None:
    assert verbalize_numbers("500\u202fmilyon") == "beş yüz milyon"
    assert verbalize_numbers("500000000000") == "beş yüz milyard"


def test_grouped_currency_percentage_ratio_and_decimal_are_verbalized() -> None:
    value = prepare_speech_text(
        "Məbləğ 1 250 000 AZN, göstərici 25%, nisbət 1/5 və əmsal 2.05-dir."
    )

    assert value == (
        "Məbləğ bir milyon iki yüz əlli min manat, göstərici iyirmi beş faiz, "
        "nisbət bir bölü beş və əmsal iki nöqtə sıfır beş-dir."
    )


def test_grouped_decimal_and_leading_zero_identifier_preserve_value() -> None:
    value = verbalize_numbers("Məbləğ 1 250 000,50 manat, kod isə 007-dir.")

    assert value == (
        "Məbləğ bir milyon iki yüz əlli min vergül əlli manat, "
        "kod isə sıfır sıfır yeddi-dir."
    )


def test_date_and_legal_locator_keep_their_semantics() -> None:
    value = verbalize_numbers("12.3.1-ci bənd 15.08.2026-cı il qüvvəyə minir.")

    assert value == (
        "on iki nöqtə üç nöqtə birinci bənd "
        "on beş avqust iki min iyirmi altıncı il qüvvəyə minir."
    )


def test_identifiers_attached_to_letters_are_not_rewritten() -> None:
    assert verbalize_numbers("Hesab ABC123 və kod 45XYZ-dir.") == (
        "Hesab ABC123 və kod 45XYZ-dir."
    )
