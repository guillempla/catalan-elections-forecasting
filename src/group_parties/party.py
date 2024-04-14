"""
Class that represents a party.
"""


class Party:
    def __init__(
        self,
        party_name: str,
        party_abbr: str,
        party_code: int,
        party_clean_name: str,
        party_clean_abbr: str,
        votes: int,
    ):
        self._name: str = party_name
        self._abbr: str = party_abbr
        self._code: int = party_code
        self._clean_name: str = party_clean_name
        self._clean_abbr: str = party_clean_abbr
        self._votes: int = votes
        self._joined: bool = False
        self.similar_parties: dict[str, "Party"] = {}
        self._participated_in: set = set()

    def __str__(self):
        return f"{self._name} ({self.code}-{self._abbr})"

    def __repr__(self):
        return f"{self._name} ({self.code}-{self._abbr})"

    def __eq__(self, other):
        return (
            self.code
            == other.code & self._name
            == other.name & self._abbr
            == other.abbr & self._votes
            == other.votes
        )

    def __lt__(self, other):
        return self._votes < other.votes

    def __le__(self, other):
        return self._votes <= other.votes

    def __ne__(self, other):
        return self._votes != other.votes

    def __gt__(self, other):
        return self._votes > other.votes

    def __ge__(self, other):
        return self._votes >= other.votes

    def __hash__(self):
        return hash(self._code)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def abbr(self):
        return self._abbr

    @abbr.setter
    def abbr(self, value):
        self._abbr = value

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    @property
    def clean_name(self):
        return self._clean_name

    @clean_name.setter
    def clean_name(self, value):
        self._clean_name = value

    @property
    def clean_abbr(self):
        return self._clean_abbr

    @clean_abbr.setter
    def clean_abbr(self, value):
        self._clean_abbr = value

    @property
    def votes(self):
        return self._votes

    @votes.setter
    def votes(self, value):
        self._votes = value

    @property
    def joined(self):
        return self._joined

    @joined.setter
    def joined(self, value):
        self._joined = value

    @property
    def similar_parties(self):
        return self._similar_parties

    @similar_parties.setter
    def similar_parties(self, value):
        self._similar_parties = value

    @property
    def participated_in(self):
        return self._participated_in

    @participated_in.setter
    def participated_in(self, value):
        self._participated_in = value

    def competed_together(self, other_party):
        return self.participated_in.intersection(other_party.participated_in)
