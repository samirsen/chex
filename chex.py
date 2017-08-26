#!/usr/bin/env python
"""
chex

Indexes chess game states from one or more PGN files
(https://en.wikipedia.org/wiki/Portable_Game_Notation) with Spotify's annoy
(https://github.com/spotify/annoy) so the user can search for game states
similar to a game state they input as well as the games in which they're
found.

Requires https://pypi.python.org/pypi/python-chess, 
https://github.com/spotify/annoy, and https://pypi.python.org/pypi/sqlitedict.
"""
import chess
from annoy import Annoyindex
from sqlitedict import SqliteDict

_help_intro = """chex is a search engine for chess game states."""

def help_formatter(prog):
    """ So formatter_class's max_help_position can be changed. """
    return argparse.HelpFormatter(prog, max_help_position=40)

def node_to_bitvector(node):
    """ Converts chess module's node to bitvector game state representation.

        node: game object of type chess.pgn.Game

        Return value: binary vector of length 768 as Python list
    """

class ChexIndex(AnnoyIndex):
    """ Manages game states from Annoy Index and SQL database. """

    def __init__(self, chex_index, id_label='FICSGamesDBGameNo',
                    first_indexed_move=10):
        """ Number of dimensions is always 8 x 8 x 12; there are 6 black piece
        types, six white piece types, and the board is 8 x 8."""
        super(ChexIndex, self).__init__(768, metric='angular')
        self.id_label = id_label
        self.first_indexed_move = first_indexed_move
        self.chex_index = chex_index
        self.chex_sql = SqliteDict(
                            os.path.join(self.chex_index, 'sqlite.idx'))

    def add_game(node):
        """ Adds game parsed by chess library to chex index.

            node: game object of type chess.pgn.Game

            No return value.
        """
        i = 0
        while not node.is_end():
            i += 1
            node = node.variations[0]
            if i < first_indexed_move: continue
            

    def save():
        super(ChexIndex, self).save(
                os.path.join(self.chex_index, 'annoy.idx')
            )
        self.chex_sql.commit()
        self.chex_sql.close()

if __name__ == '__main__':
    # Print file's docstring if -h is invoked
    parser = argparse.ArgumentParser(description=_help_intro, 
                formatter_class=help_formatter)
    subparsers = parser.add_subparsers(help=(
                'subcommands; add "-h" or "--help" '
                'after a subcommand for its parameters'),
                dest='subparser_name'
            )
    index_parser = subparsers.add_parser(
                            'index',
                            help='creates index of chess game states'
                        )
    search_parser = subparsers.add_parser(
                            'search',
                            help=('searches for chess game states similar to '
                                  'those input by user')
                        )
    index_parser.add_argument('-f', '--first-indexed-move',
            metavar='<int>', type=int, required=False,
            default=10,
            help=('indexes only those game states at least this many moves '
                  'into a given game')
        )
    index_parser.add_argument('-p', '--pgn', metavar='<file(s)>', nargs='+',
            required=True, type=str,
            help='space-separated list of PGNs to index'
        )
    index_parser.add_argument('-i', '--id-label', metavar='<str>',
            required=False, type=str,
            default='FICSGamesDBGameNo',
            help='game ID label from metadata in PGN files'
        )
    index_parser.add_argument('-x', '--chex-index', metavar='<dir>',
            required=True, type=str,
            help='directory in which to store chex index files'
        )
    search_parser.add_argument('-p', '--pgn', metavar='<file>',
            required=True, type=str,
            help='PGN describing game with state to search for')
    search_parser.add_argument('-m', '--move', metavar='<int>',
            required=True, type=int,
            help='move number from PGN corresponding to state to search for')
    index_parser.add_argument('-x', '--chex-index', metavar='<dir>',
            required=True, type=str,
            help='chex index directory'
        )
    args = parser.parse_args()
