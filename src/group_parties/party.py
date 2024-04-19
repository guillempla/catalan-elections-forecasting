"""
Class that represents a party.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Party:
    def __init__(
        self,
        party_name: str,
        party_abbr: str,
        party_code: int,
        party_clean_name: str,
        party_clean_abbr: str,
        party_color: str,
        votes: int,
    ):
        self._name: str = party_name
        self._abbr: str = party_abbr
        self._code: int = party_code
        self._clean_name: str = party_clean_name
        self._clean_abbr: str = party_clean_abbr
        self._color: str = party_color
        self._votes: int = votes
        self._joined: bool = False
        self._joined_code: int = None
        self._similar_parties: dict[str, "Party"] = {}
        self._participated_in: set = set()
        self._group_participated_in: set = set()
        self._grouped_parties: set = set()

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

    def __neg__(self):
        return -self._votes

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
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

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
    def joined_code(self):
        return self._joined_code

    @joined_code.setter
    def joined_code(self, value):
        self._joined = True
        self._joined_code = value

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
        self.group_participated_in = value.copy()

    @property
    def group_participated_in(self):
        return self._group_participated_in

    @group_participated_in.setter
    def group_participated_in(self, value):
        self._group_participated_in = value

    @property
    def grouped_parties(self):
        return self._grouped_parties

    @grouped_parties.setter
    def grouped_parties(self, value):
        self._grouped_parties = value

    def _join_participated_in(self, other_party):
        self.group_participated_in = self.group_participated_in.union(
            other_party.group_participated_in
        )

    def _competed_together(self, other_party):
        return bool(self.participated_in.intersection(other_party.participated_in))

    def _group_competed_together(self, other_party):
        return bool(
            self.group_participated_in.intersection(other_party.group_participated_in)
        )

    def _add_grouped_party(self, party):
        self._grouped_parties.add(party)

    def add_similar_party(self, party):
        self._similar_parties[party.code] = party

    def join_parties(self, other_party: "Party", verbose=False) -> bool:
        if other_party.joined:
            if verbose:
                logging.warning(
                    "Party %s has already been joined to party %s. It will not be joined again.",
                    other_party,
                    other_party.joined_code,
                )
            return False
        if self.joined_code is not None and self.joined_code != self.code:
            if verbose:
                logging.warning(
                    "Party %s has already been joined to party %s. It will not be joined again.",
                    self,
                    self.joined_code,
                )
            return False
        if self._group_competed_together(other_party):
            if verbose:
                logging.warning(
                    "Parties (or group) %s and %s competed together. They will not be joined.",
                    self,
                    other_party,
                )
            return False

        self._join_participated_in(other_party)
        other_party.joined_code = self.code
        if self.joined_code is None:
            self.joined_code = self.code
        self._add_grouped_party(other_party)
        return True
