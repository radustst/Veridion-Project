"""Deterministic geography resolution.

Geography is the single most common hard constraint in these queries and the
one an LLM is most likely to answer inconsistently across runs -- ask twice
whether Finland is "Scandinavia" and you can get two answers. Resolving region
words from a static table makes the answer free, instant and identical every
time, and leaves the LLM to do only the part that actually needs judgement.

The planner is still allowed to emit raw ISO-2 codes for anything not in this
table; this module only guarantees that the common region words are stable.
"""
from __future__ import annotations

import re

# Region words -> ISO-3166-1 alpha-2. Deliberately explicit rather than clever.
NORDICS = {"se", "no", "dk", "fi", "is"}
SCANDINAVIA = {"se", "no", "dk"}

EU_27 = {
    "at", "be", "bg", "hr", "cy", "cz", "dk", "ee", "fi", "fr", "de", "gr",
    "hu", "ie", "it", "lv", "lt", "lu", "mt", "nl", "pl", "pt", "ro", "sk",
    "si", "es", "se",
}
EUROPE = EU_27 | {
    "gb", "uk", "ch", "no", "is", "li", "mc", "sm", "ad", "va", "rs", "ba",
    "me", "mk", "al", "md", "ua", "by", "ru", "tr", "ge", "am", "az", "xk",
}
DACH = {"de", "at", "ch"}
BENELUX = {"be", "nl", "lu"}
IBERIA = {"es", "pt"}
BALTICS = {"ee", "lv", "lt"}
UK_IE = {"gb", "ie"}
NORTH_AMERICA = {"us", "ca", "mx"}
LATAM = {"mx", "br", "ar", "cl", "co", "pe", "uy", "ve", "ec", "bo", "py", "cr", "pa"}
APAC = {
    "cn", "jp", "kr", "in", "au", "nz", "sg", "my", "th", "id", "ph", "vn",
    "tw", "hk", "bd", "pk", "lk",
}
MENA = {"ae", "sa", "qa", "kw", "bh", "om", "eg", "ma", "tn", "dz", "jo", "lb", "il", "iq", "ir"}
AFRICA = {
    "za", "ng", "ke", "eg", "ma", "gh", "tz", "ug", "et", "dz", "tn", "sn",
    "ci", "cm", "zm", "zw", "rw", "mz", "ao",
}

REGION_ALIASES: dict[str, set[str]] = {
    "europe": EUROPE,
    "european": EUROPE,
    "european union": EU_27,
    "eu": EU_27,
    "eea": EU_27 | {"no", "is", "li"},
    "western europe": {"fr", "de", "nl", "be", "lu", "at", "ch", "gb", "ie", "es", "pt", "it"},
    "eastern europe": {"pl", "cz", "sk", "hu", "ro", "bg", "hr", "si", "ee", "lv", "lt", "ua", "md", "rs"},
    "northern europe": NORDICS | {"ee", "lv", "lt", "gb", "ie"},
    "southern europe": {"es", "pt", "it", "gr", "hr", "si", "mt", "cy"},
    "central europe": {"de", "at", "ch", "pl", "cz", "sk", "hu", "si"},
    "scandinavia": SCANDINAVIA,
    "scandinavian": SCANDINAVIA,
    "nordics": NORDICS,
    "nordic": NORDICS,
    "nordic countries": NORDICS,
    "dach": DACH,
    "benelux": BENELUX,
    "iberia": IBERIA,
    "iberian peninsula": IBERIA,
    "baltics": BALTICS,
    "baltic states": BALTICS,
    "british isles": UK_IE,
    "north america": NORTH_AMERICA,
    "latin america": LATAM,
    "south america": {"br", "ar", "cl", "co", "pe", "uy", "ve", "ec", "bo", "py"},
    "apac": APAC,
    "asia pacific": APAC,
    "asia-pacific": APAC,
    "southeast asia": {"sg", "my", "th", "id", "ph", "vn", "kh", "la", "mm", "bn"},
    "middle east": {"ae", "sa", "qa", "kw", "bh", "om", "jo", "lb", "il", "iq", "ir", "ye", "sy"},
    "mena": MENA,
    "gcc": {"ae", "sa", "qa", "kw", "bh", "om"},
    "africa": AFRICA,
    "oceania": {"au", "nz", "fj", "pg"},
    "global": set(),  # no constraint
    "worldwide": set(),
    "anywhere": set(),
}

COUNTRY_NAMES: dict[str, str] = {
    "afghanistan": "af", "albania": "al", "algeria": "dz", "andorra": "ad", "angola": "ao",
    "argentina": "ar", "armenia": "am", "australia": "au", "austria": "at", "azerbaijan": "az",
    "bahrain": "bh", "bangladesh": "bd", "belarus": "by", "belgium": "be", "bolivia": "bo",
    "bosnia": "ba", "bosnia and herzegovina": "ba", "brazil": "br", "brunei": "bn",
    "bulgaria": "bg", "cambodia": "kh", "cameroon": "cm", "canada": "ca", "chile": "cl",
    "china": "cn", "colombia": "co", "costa rica": "cr", "croatia": "hr", "cyprus": "cy",
    "czech republic": "cz", "czechia": "cz", "denmark": "dk", "ecuador": "ec", "egypt": "eg",
    "estonia": "ee", "ethiopia": "et", "finland": "fi", "france": "fr", "georgia": "ge",
    "germany": "de", "ghana": "gh", "greece": "gr", "hong kong": "hk", "hungary": "hu",
    "iceland": "is", "india": "in", "indonesia": "id", "iran": "ir", "iraq": "iq",
    "ireland": "ie", "israel": "il", "italy": "it", "ivory coast": "ci", "japan": "jp",
    "jordan": "jo", "kazakhstan": "kz", "kenya": "ke", "kuwait": "kw", "latvia": "lv",
    "lebanon": "lb", "liechtenstein": "li", "lithuania": "lt", "luxembourg": "lu",
    "malaysia": "my", "malta": "mt", "mexico": "mx", "moldova": "md", "monaco": "mc",
    "montenegro": "me", "morocco": "ma", "mozambique": "mz", "myanmar": "mm",
    "netherlands": "nl", "holland": "nl", "new zealand": "nz", "nigeria": "ng",
    "north macedonia": "mk", "macedonia": "mk", "norway": "no", "oman": "om",
    "pakistan": "pk", "panama": "pa", "papua new guinea": "pg", "paraguay": "py",
    "peru": "pe", "philippines": "ph", "poland": "pl", "portugal": "pt", "qatar": "qa",
    "romania": "ro", "russia": "ru", "russian federation": "ru", "rwanda": "rw",
    "san marino": "sm", "saudi arabia": "sa", "senegal": "sn", "serbia": "rs",
    "singapore": "sg", "slovakia": "sk", "slovenia": "si", "south africa": "za",
    "south korea": "kr", "korea": "kr", "spain": "es", "sri lanka": "lk", "sweden": "se",
    "switzerland": "ch", "syria": "sy", "taiwan": "tw", "tanzania": "tz", "thailand": "th",
    "tunisia": "tn", "turkey": "tr", "uganda": "ug", "ukraine": "ua",
    "united arab emirates": "ae", "uae": "ae", "united kingdom": "gb", "uk": "gb",
    "britain": "gb", "great britain": "gb", "england": "gb", "scotland": "gb",
    "wales": "gb", "northern ireland": "gb", "united states": "us",
    "united states of america": "us", "usa": "us", "us": "us", "america": "us",
    "uruguay": "uy", "uzbekistan": "uz", "venezuela": "ve", "vietnam": "vn",
    "yemen": "ye", "zambia": "zm", "zimbabwe": "zw",
}

# ISO-2 codes the dataset uses that need normalising to a canonical form.
CODE_NORMALISE = {"uk": "gb", "el": "gr"}


def normalise_code(code: str | None) -> str | None:
    if not code:
        return None
    c = code.strip().lower()
    return CODE_NORMALISE.get(c, c) if len(c) == 2 else None


def resolve(tokens: list[str]) -> set[str]:
    """Expand a mixed list of region words, country names and ISO-2 codes.

    Unknown tokens are dropped rather than guessed -- a token we cannot resolve
    must not silently narrow the search to nothing.
    """
    out: set[str] = set()
    for raw in tokens:
        if not raw:
            continue
        t = re.sub(r"[^a-z\s-]", "", str(raw).strip().lower()).strip()
        if not t:
            continue
        if t in REGION_ALIASES:
            out |= REGION_ALIASES[t]
        elif t in COUNTRY_NAMES:
            out.add(COUNTRY_NAMES[t])
        elif len(t) == 2:
            out.add(CODE_NORMALISE.get(t, t))
        elif len(t) == 3:
            # tolerate the odd alpha-3 the planner may emit
            alpha3 = {"usa": "us", "gbr": "gb", "deu": "de", "fra": "fr", "che": "ch",
                      "swe": "se", "nor": "no", "dnk": "dk", "fin": "fi", "rou": "ro",
                      "esp": "es", "ita": "it", "nld": "nl", "chn": "cn", "jpn": "jp"}
            if t in alpha3:
                out.add(alpha3[t])
    return out


def extract_from_text(text: str) -> tuple[set[str], str]:
    """Best-effort geography scan of raw query text.

    A safety net: if the planner returns no geography but the query plainly
    names one, apply the constraint anyway. Matches the longest phrase first so
    "united states" wins over the bare token "us", and stops at the first hit
    so "renewable energy equipment manufacturers in Scandinavia" resolves to
    Scandinavia rather than also dragging in every country word in the string.
    """
    low = " " + re.sub(r"[^a-z\s]", " ", text.lower()) + " "
    lookup: dict[str, set[str]] = {k: set(v) for k, v in REGION_ALIASES.items()}
    for name, code in COUNTRY_NAMES.items():
        lookup.setdefault(name, {code})

    for phrase in sorted(lookup, key=len, reverse=True):
        if len(phrase) <= 2:
            continue  # too ambiguous to pattern-match in free text
        if " " + phrase + " " in low:
            return set(lookup[phrase]), phrase
    return set(), ""
