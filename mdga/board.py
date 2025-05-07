from dataclasses import dataclass
import enum
from typing import Callable, Optional

MIN_PLAYERS: int = 2
MAX_PLAYERS: int = 4
PIECES_PER_PLAYER: int = 4

MIN_ROLL: int = 1
MAX_ROLL: int = 6

# Fields on which each index places their piece once leaving the home area
FIELD_ENTRANCE: list[int] = [0, 10, 20, 30]

# Fields on which each index starts entering their target area
TARGET_ENTRANCE: list[int] = [39, 9, 19, 29]

# How many transit field count
TRANSIT_FIELDS: int = 40


class PieceState(enum.StrEnum):
    home = enum.auto()
    transit = enum.auto()
    target = enum.auto()


@dataclass
class Piece:
    id: int
    position: Optional[int] = None

    def __str__(self) -> str:
        return str(self.id)

    def __hash__(self) -> int:
        return hash((self.id, self.position))

    @property
    def state(self) -> PieceState:
        if self.position is None:
            return PieceState.home

        if self.position < 0:
            assert self.position >= -PIECES_PER_PLAYER
            return PieceState.target

        assert self.position < TRANSIT_FIELDS
        return PieceState.transit


class InvalidMoveError(RuntimeError):
    pass


class Board:
    pieces: tuple[Piece, ...]

    def __init__(self) -> None:
        self.pieces = tuple(Piece(id=index // MAX_PLAYERS) for index in range(MAX_PLAYERS * PIECES_PER_PLAYER))

    def print(self, view_id: int, log: Callable[[str], None] = print) -> None:
        home = ""
        for _ in range(MAX_PLAYERS):
            fields = list("-" * PIECES_PER_PLAYER)
            for index, piece in enumerate(self.filter(id=view_id, state=PieceState.home)):
                assert piece.position is None
                fields[index] = str(piece)

            section = "".join(fields)
            home += section.ljust(TRANSIT_FIELDS // MAX_PLAYERS)
            view_id = (view_id + 1) % MAX_PLAYERS

        fields = list("*" * TRANSIT_FIELDS)
        for piece in self.filter(state=PieceState.transit):
            assert piece.position is not None
            # We make use of negative indexing to index from the end when going negative
            fields[piece.position - FIELD_ENTRANCE[view_id]] = str(piece)
        transit = "".join(fields)

        target = ""
        for _ in range(MAX_PLAYERS):
            fields = list("~" * PIECES_PER_PLAYER)
            for piece in self.filter(id=view_id, state=PieceState.target):
                assert piece.position is not None
                fields[-(piece.position + 1)] = str(piece)

            section = "".join(fields)
            target += section.ljust(TRANSIT_FIELDS // MAX_PLAYERS)
            view_id = (view_id + 1) % MAX_PLAYERS

        log(f"home:    {home}")
        log(f"transit: {transit}")
        log(f"target:  {target}")

        pass

    def assert_uniqueness(self) -> None:
        occupied: set[int] = set()

        for piece in self.pieces:
            # Only pieces that are in transit have any meaningful position that they can share with other pieces
            if piece.state != PieceState.transit:
                continue

            assert piece.position is not None
            assert piece.position not in occupied
            occupied.add(piece.position)

    def simulate_move(self, piece: Piece, roll: int) -> int:
        assert piece in self.pieces
        assert MIN_ROLL <= roll <= MAX_ROLL

        if piece.position is None:
            if roll != MAX_ROLL:
                raise InvalidMoveError("Attempting to move out of home without a 6")

            newpos = FIELD_ENTRANCE[piece.id]

        elif piece.position < 0:
            newpos = piece.position - roll

        else:
            newpos = piece.position

            for _ in range(roll):
                if newpos == TARGET_ENTRANCE[piece.id]:
                    newpos = -1  # The target fields are indexed as negative numbers

                elif newpos < 0:
                    newpos -= 1

                else:
                    newpos = (newpos + 1) % TRANSIT_FIELDS

        if newpos < -PIECES_PER_PLAYER:
            raise InvalidMoveError("Attempting to move too deep into the target area")

        if any(self.filter(id=piece.id, position=newpos)):
            raise InvalidMoveError("Attempting to move onto our own piece")

        return newpos

    def score_move(self, piece: Piece, roll: int) -> float:
        # For now just take the distance of the target and multiply it by a constant
        KNOCKOUT_MULTIPLIER = 10
        # It is less likely that we get knocked out, but we still want to account for it,
        # so score a knockout and 3 pieces knocking us out as roughly equal
        DANGER_MULTIPLIER = 3
        # We only get out of the target area with a MAX_ROLL
        MOVE_FROM_HOME_SCORE = MAX_ROLL
        # If we can better fill up the end of the target area
        MOVE_IN_TARGET_SCORE = 2
        # If we can move into the target area we should prefer that over moving inside it
        MOVE_TO_TARGET_SCORE = MOVE_IN_TARGET_SCORE + PIECES_PER_PLAYER * 2

        def position_danger_count(position: int) -> int:
            if position < 0:
                return 0

            danger = 0
            for i in range(MIN_ROLL, MAX_ROLL + 1):
                for other in self.pieces:
                    if other.id == piece.id:
                        continue

                    try:
                        if self.simulate_move(other, i) == position:
                            danger += 1

                    except InvalidMoveError:
                        pass

            return danger

        distance = self.distance(piece)

        # The baseline value of a piece is how it has so far gone
        score = distance

        if (simulated_position := self.simulate_move(piece, roll)) >= 0:

            for other in self.filter(position=simulated_position):
                score += self.distance(other) * KNOCKOUT_MULTIPLIER

            # If the simulated position is under threat we want to discourage moving there
            score -= distance * position_danger_count(simulated_position) * DANGER_MULTIPLIER

            if piece.position is None:
                score += MOVE_FROM_HOME_SCORE
            else:
                # If the current position is under threat we want to encourage moving from it
                score += distance * position_danger_count(piece.position) * DANGER_MULTIPLIER

        else:
            assert piece.position is not None
            score += MOVE_IN_TARGET_SCORE if piece.position < 0 else MOVE_TO_TARGET_SCORE

        return score

    def move(self, piece: Piece, roll: int) -> None:
        newpos = self.simulate_move(piece, roll)

        # We are only able to knock out pieces that are in transit. Any pieces on home or target fields are protected
        # TODO: We probably should also ensure that at maximum a single piece is found at this position
        for other in self.filter(position=newpos, state=PieceState.transit):
            # If the simulated move returns successfully we can assume that we aren't moving on to one of our own pieces
            # so we can assume that the filter will not return any with the current piece ID.
            assert other.id != piece.id
            other.position = None

        piece.position = newpos
        self.assert_uniqueness()

    def distance(self, piece: Piece) -> int:
        if piece.position is None:
            return 0

        if piece.position < 0:
            # We arrived in the target fields and since they are negative we need to add them to the total
            # transit field count, so we have to subtract the negative value it contains
            return TRANSIT_FIELDS - piece.position

        # Since we have a distance of 0 when we are in the home fields we start with a distance of 1 when we are on the
        # entrance field. And for each field that we aren't on after that we increase the distance by 1
        rv = 1
        pos = FIELD_ENTRANCE[piece.id]
        while pos != piece.position:
            rv += 1
            pos = (pos + 1) % TRANSIT_FIELDS

        return rv

    def filter(
        self,
        *,
        id: Optional[int] = None,
        position: Optional[int] = None,
        state: Optional[PieceState] = None,
    ) -> tuple[Piece, ...]:
        pieces = self.pieces

        if id is not None:
            pieces = filter(lambda piece: piece.id == id, pieces)

        if position is not None:
            pieces = filter(lambda piece: piece.position == position, pieces)

        if state is not None:
            pieces = filter(lambda piece: piece.state == state, pieces)

        return tuple(pieces)
