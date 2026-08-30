"""Deterministic Azerbaijani number verbalization for legal-answer TTS."""

from __future__ import annotations

import re


_DIGITS = (
    "sıfır",
    "bir",
    "iki",
    "üç",
    "dörd",
    "beş",
    "altı",
    "yeddi",
    "səkkiz",
    "doqquz",
)
_TENS = (
    "",
    "on",
    "iyirmi",
    "otuz",
    "qırx",
    "əlli",
    "altmış",
    "yetmiş",
    "səksən",
    "doxsan",
)
_SCALES = (
    "",
    "min",
    "milyon",
    "milyard",
    "trilyon",
    "kvadrilyon",
    "kvintilyon",
)
_MONTHS = (
    "",
    "yanvar",
    "fevral",
    "mart",
    "aprel",
    "may",
    "iyun",
    "iyul",
    "avqust",
    "sentyabr",
    "oktyabr",
    "noyabr",
    "dekabr",
)
_VOWELS = frozenset("aıoueəiöü")
_ORDINAL_SUFFIXES = {
    "a": "ıncı",
    "ı": "ıncı",
    "e": "inci",
    "ə": "inci",
    "i": "inci",
    "o": "uncu",
    "u": "uncu",
    "ö": "üncü",
    "ü": "üncü",
}

_GROUP_SPACE = "    "
_GROUP_SPACE_TRANSLATION = str.maketrans(
    {"\u00a0": " ", "\u202f": " ", "\u2009": " "}
)
_NUMBER_TOKEN = rf"(?:\d{{1,3}}(?:[{_GROUP_SPACE}]\d{{3}})+|\d+)"
_ORDINAL_SUFFIX = r"(?:cı|ci|cu|cü)"

_DATE_RE = re.compile(
    rf"(?<!\w)(?P<day>0?[1-9]|[12]\d|3[01])[./]"
    rf"(?P<month>0?[1-9]|1[0-2])[./](?P<year>\d{{4}})"
    rf"(?P<ordinal>-{_ORDINAL_SUFFIX})?(?!\w)",
    re.IGNORECASE,
)
_ORDINAL_LOCATOR_RE = re.compile(
    rf"(?<!\w)(?P<parts>\d+(?:\.\d+)+)-{_ORDINAL_SUFFIX}(?!\w)",
    re.IGNORECASE,
)
_LOCATOR_RE = re.compile(r"(?<!\w)(?P<parts>\d+(?:\.\d+){2,})(?!\w)")
_RATIO_RE = re.compile(
    rf"(?<!\w)(?P<numerator>{_NUMBER_TOKEN})\s*/\s*"
    rf"(?P<denominator>{_NUMBER_TOKEN})(?!\w)"
)
_ORDINAL_RE = re.compile(
    rf"(?<!\w)(?P<number>{_NUMBER_TOKEN})-{_ORDINAL_SUFFIX}(?!\w)",
    re.IGNORECASE,
)
_SPACE_GROUPED_INTEGER_RE = re.compile(
    rf"(?<![\w.,])(?P<number>[+-]?\d{{1,3}}(?:[{_GROUP_SPACE}]\d{{3}})+)"
    rf"(?![\w.,])"
)
# Two or more commas are unambiguously group separators. A single comma is
# treated as a decimal separator in Azerbaijani text.
_COMMA_GROUPED_INTEGER_RE = re.compile(
    r"(?<![\w,])(?P<number>[+-]?\d{1,3}(?:,\d{3}){2,})(?![\w,])"
)
_DECIMAL_RE = re.compile(
    rf"(?<![\w.,])(?P<sign>[+-]?)(?P<integer>{_NUMBER_TOKEN})"
    rf"(?P<separator>[.,])(?P<fraction>\d+)(?![\w.,])"
)
_INTEGER_RE = re.compile(
    r"(?<![\w])(?P<sign>[+-]?)(?P<number>\d+)(?![\w])"
)


def integer_to_azerbaijani(value: int) -> str:
    """Render a signed integer using Azerbaijani short-scale number names."""

    if value == 0:
        return _DIGITS[0]

    sign = "mənfi " if value < 0 else ""
    number = abs(value)
    groups: list[int] = []
    while number:
        number, group = divmod(number, 1_000)
        groups.append(group)

    if len(groups) > len(_SCALES):
        # Extremely large identifiers are safer read digit-by-digit than with
        # an unsupported or invented scale name.
        digits = " ".join(_DIGITS[int(char)] for char in str(abs(value)))
        return sign + digits

    words: list[str] = []
    for scale_index in range(len(groups) - 1, -1, -1):
        group = groups[scale_index]
        if group == 0:
            continue
        if not (scale_index == 1 and group == 1):
            words.extend(_three_digits_to_words(group))
        if scale_index:
            words.append(_SCALES[scale_index])
    return sign + " ".join(words)


def integer_to_ordinal_azerbaijani(value: int) -> str:
    """Render a non-negative integer as an Azerbaijani ordinal word."""

    if value < 0:
        return f"mənfi {integer_to_ordinal_azerbaijani(abs(value))}"
    cardinal = integer_to_azerbaijani(value)
    last_vowel = next((char for char in reversed(cardinal) if char in _VOWELS), None)
    if last_vowel is None:  # Defensive; every supported number word has a vowel.
        return cardinal
    suffix = _ORDINAL_SUFFIXES[last_vowel]
    if cardinal[-1] in _VOWELS:
        cardinal = cardinal[:-1]
    return cardinal + suffix


def verbalize_numbers(text: str) -> str:
    """Convert numeric expressions to speech-safe Azerbaijani words.

    Specific legal forms are handled before generic integers so dates,
    locators, ordinals, ratios and decimals retain their intended meaning.
    """

    value = text.translate(_GROUP_SPACE_TRANSLATION)
    value = _DATE_RE.sub(_replace_date, value)
    value = _ORDINAL_LOCATOR_RE.sub(_replace_ordinal_locator, value)
    value = _LOCATOR_RE.sub(_replace_locator, value)
    value = _RATIO_RE.sub(_replace_ratio, value)
    value = _ORDINAL_RE.sub(_replace_ordinal, value)
    value = _DECIMAL_RE.sub(_replace_decimal, value)
    value = _SPACE_GROUPED_INTEGER_RE.sub(_replace_grouped_integer, value)
    value = _COMMA_GROUPED_INTEGER_RE.sub(_replace_grouped_integer, value)
    return _INTEGER_RE.sub(_replace_integer, value)


def _three_digits_to_words(value: int) -> list[str]:
    words: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        if hundreds > 1:
            words.append(_DIGITS[hundreds])
        words.append("yüz")
    tens, units = divmod(remainder, 10)
    if tens:
        words.append(_TENS[tens])
    if units:
        words.append(_DIGITS[units])
    return words


def _token_to_int(token: str) -> int:
    compact = re.sub(rf"[{_GROUP_SPACE},]", "", token)
    return int(compact)


def _fraction_to_words(fraction: str) -> str:
    leading_zero_count = len(fraction) - len(fraction.lstrip("0"))
    words = [_DIGITS[0]] * leading_zero_count
    remainder = fraction[leading_zero_count:]
    if remainder:
        words.append(integer_to_azerbaijani(int(remainder)))
    return " ".join(words or [_DIGITS[0]])


def _integer_token_to_words(token: str) -> str:
    compact = re.sub(rf"[{_GROUP_SPACE},]", "", token)
    if len(compact) > 1 and compact.startswith("0"):
        return " ".join(_DIGITS[int(char)] for char in compact)
    return integer_to_azerbaijani(int(compact))


def _replace_date(match: re.Match[str]) -> str:
    day = integer_to_azerbaijani(int(match.group("day")))
    month = _MONTHS[int(match.group("month"))]
    year_value = int(match.group("year"))
    year = (
        integer_to_ordinal_azerbaijani(year_value)
        if match.group("ordinal")
        else integer_to_azerbaijani(year_value)
    )
    return f"{day} {month} {year}"


def _replace_ordinal_locator(match: re.Match[str]) -> str:
    parts = match.group("parts").split(".")
    words = [integer_to_azerbaijani(int(part)) for part in parts[:-1]]
    words.append(integer_to_ordinal_azerbaijani(int(parts[-1])))
    return " nöqtə ".join(words)


def _replace_locator(match: re.Match[str]) -> str:
    return " nöqtə ".join(
        integer_to_azerbaijani(int(part))
        for part in match.group("parts").split(".")
    )


def _replace_ratio(match: re.Match[str]) -> str:
    numerator = integer_to_azerbaijani(_token_to_int(match.group("numerator")))
    denominator = integer_to_azerbaijani(
        _token_to_int(match.group("denominator"))
    )
    return f"{numerator} bölü {denominator}"


def _replace_ordinal(match: re.Match[str]) -> str:
    return integer_to_ordinal_azerbaijani(_token_to_int(match.group("number")))


def _replace_grouped_integer(match: re.Match[str]) -> str:
    return integer_to_azerbaijani(_token_to_int(match.group("number")))


def _replace_decimal(match: re.Match[str]) -> str:
    integer = _integer_token_to_words(match.group("integer"))
    fraction = _fraction_to_words(match.group("fraction"))
    separator = "nöqtə" if match.group("separator") == "." else "vergül"
    sign = {"-": "mənfi ", "+": "müsbət "}.get(match.group("sign"), "")
    return f"{sign}{integer} {separator} {fraction}"


def _replace_integer(match: re.Match[str]) -> str:
    sign = match.group("sign")
    words = _integer_token_to_words(match.group("number"))
    if sign == "-":
        return f"mənfi {words}"
    return f"müsbət {words}" if sign == "+" else words
