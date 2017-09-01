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
import chess.pgn
import struct
import binascii
import argparse
import errno
import os
import sys
import time
from annoy import AnnoyIndex
from sqlitedict import SqliteDict

_help_intro = """chex is a search engine for chess game states."""

def help_formatter(prog):
    """ So formatter_class's max_help_position can be changed. """
    return argparse.HelpFormatter(prog, max_help_position=40)

# For bitvector conversion
_offsets = {
    'p' : 0,
    'P' : 1,
    'n' : 2,
    'N' : 3,
    'b' : 4,
    'B' : 5,
    'k' : 6,
    'K' : 7,
    'r' : 8,
    'R' : 9,
    'q' : 10,
    'Q' : 11,
}

_reverse_offsets = { value : key for key, value in _offsets.items() }

_reverse_colors_offsets = {
    'p' : 1,
    'P' : 0,
    'n' : 3,
    'N' : 2,
    'b' : 5,
    'B' : 4,
    'k' : 7,
    'K' : 6,
    'r' : 9,
    'R' : 8,
    'q' : 11,
    'Q' : 10,
}

def board_to_bitvector(board):
    """ Converts chess module's board to bitvector game state representation.

        node: game object of type chess.pgn.Game

        Return value: binary vector of length 768 as Python list
    """
    bitvector = [0 for _ in xrange(768)]
    for i in xrange(64):
        try:
            bitvector[i*12 + _offsets[board.piece_at(i).symbol()]] = 1
        except AttributeError:
            pass
    return bitvector

def bitvector_to_board(bitvector):
    """ Converts bitvector to board.

        TODO: unit test.

        bitvector: iterable of 768 1s and 0s

        Return value: chess.Board representation of bitvector
    """
    fen = [[] for _ in xrange(8)]
    for i in xrange(8):
        streak = 0
        for j in xrange(8):
            segment = 12*(8*i + j)
            piece = None
            for offset in xrange(12):
                if bitvector[segment + offset]:
                    piece = _reverse_offsets[offset]
            if piece is not None:
                if streak: fen[i].append(str(streak))
                fen[i].append(piece)
                streak = 0
            else:
                streak += 1
                if j == 7: fen[i].append(str(streak))
    return chess.Board(
            '/'.join([''.join(row) for row in fen][::-1]) + ' w KQkq - 0 1'
        )

def invert_board(board):
    """ This function computes bitvector of given position but with inverted colors. """
    inversevector = [0 for _ in xrange(768)]
    for i in xrange(64):
        try:
            inversevector[i * 12 + _reverse_colors_offsets[board.piece_at(i).symbol()]] = 1
        except AttributeError:
            pass
    return inversevector

def flip_board(board):
    """ This function computes bitvector of the mirror image of a given position. """
    flipvector = [0 for _ in xrange(768)]
    for i in range(8):
        for j in range(8):
            try:
                flipvector[12*(8*i + 7 - j) + _offsets[board.piece_at(8*i + j).symbol()]] = 1
            except AttributeError:
                pass

    return flipvector

def reverse_and_flip(board):
    """ This function computes the bitvector after flipping a given position and reversing the colors. """
    reversevector = [0 for _ in xrange(768)]
    for i in range(8):
        for j in range(8):
            try:
                reversevector[12*(8*i + 7 - j) + _reverse_colors_offsets[board.piece_at(8*i + j).symbol()]] = 1
            except AttributeError:
                pass

    return reversevector


class ChexIndex(AnnoyIndex):
    """ Manages game states from Annoy Index and SQL database. """

    #TODO: Compute ASCII of B, I(B), F(B) and I(F(B)) and only store min(ASCII's) into the chex index.

    def __init__(self, chex_index, id_label='FICSGamesDBGameNo',
                    first_indexed_move=10, n_trees=200):
        """ Number of dimensions is always 8 x 8 x 12; there are 6 black piece
        types, six white piece types, and the board is 8 x 8."""
        super(ChexIndex, self).__init__(768, metric='angular')
        self.id_label = id_label
        self.first_indexed_move = first_indexed_move
        self.chex_index = chex_index
        try:
            os.makedirs(self.chex_index)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.chex_sql = SqliteDict(
                            os.path.join(self.chex_index, 'sqlite.idx'))
        self.node_id = 0
        self.n_trees = n_trees

    def add_game(self, node):
        """ Adds game parsed by chess library to chex index.

            node: game object of type chess.pgn.Game

            Return value: 0 if game added successfully, else 1
        """
        if node is None:
            return 1

        game_id = node.headers[self.id_label]
        move_number = 0
        for move_number in xrange(self.first_indexed_move - 1):
            try:
                node = node.variations[0]
            except IndexError:
                # Too few moves to index
                return 0

        while True:
            move_number += 1

            bitvector = board_to_bitvector(node.board())
            inversevector = invert_board(node.board())
            flipvector = flip_board(node.board())
            reversevector = reverse_and_flip(node.board())

            # Store as ASCII
            # to_unhexlify = '%x' % int(''.join(map(str, bitvector)), 2)

            to_unhexlify = min(('%x' % int(''.join(map(str, bitvector)), 2)), ('%x' % int(''.join(map(str, inversevector)), 2)),
                               ('%x' % int(''.join(map(str, flipvector)), 2)), ('%x' % int(''.join(map(str, reversevector)), 2)))
            try:
                key = binascii.unhexlify(to_unhexlify)
            except TypeError:
                key = binascii.unhexlify('0' + to_unhexlify)
            if key in self.chex_index:
                self.chex_sql[key] = self.chex_sql[key] + [
                                                        (game_id, move_number)
                                                    ]
            else:
                self.chex_sql[key] = [(game_id, move_number)]
                self.add_item(self.node_id, bitvector)
                self.node_id += 1
            if node.is_end(): break
            node = node.variations[0]

        return 0

    def save(self):
        self.build(self.n_trees)
        super(ChexIndex, self).save(
                os.path.join(self.chex_index, 'annoy.idx')
            )
        self.chex_sql.commit()
        self.chex_sql.close()

class ChexSearch(object):
    """ Searches Chex index for game states and associated games. """

    #TODO: Compute B, I(B), F(B) and I(F(B)) and search these in chex index. Combine results.

    def __init__(self, chex_index, results=10, search_k=40):
        self.chex_index = chex_index
        self.results = results
        self.search_k = search_k
        self.annoy_index = AnnoyIndex(768, metric='angular')
        self.annoy_index.load(os.path.join(self.chex_index, 'annoy.idx'))
        self.chex_sql = SqliteDict(
                            os.path.join(self.chex_index, 'sqlite.idx'))

    def search(self, board):
        """ Searches for board.

            board: game object of type chess.Board

            Return value: [
                (board, similarity score, [(game_id, move number), ...]), ...]
        """

        symmetrical_boards = [board_to_bitvector(board), invert_board(board), flip_board(board), reverse_and_flip(board)]

        results = []
        for bitvector in symmetrical_boards:
            for annoy_id, similarity in zip(
                                *self.annoy_index.get_nns_by_vector(
                                        bitvector, self.results,
                                        include_distances=True
                            )):
                # Recompute ASCII key
                bitvector = self.annoy_index.get_item_vector(annoy_id)
                to_unhexlify = '%x' % int(
                                        ''.join(map(str, map(int, bitvector))), 2)
                try:
                    key = binascii.unhexlify(to_unhexlify)
                except TypeError:
                    key = binascii.unhexlify('0' + to_unhexlify)
                results.append((bitvector_to_board(bitvector), similarity,
                                self.chex_sql[key]))
        return results

    def close(self):
        del self.annoy_index

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
    index_parser.add_argument('-p', '--pgns', metavar='<files>', nargs='+',
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
    # Test various values!
    index_parser.add_argument('--n-trees', metavar='<int>', type=int,
            required=False,
            default=200,
            help='number of annoy trees'
        )
    search_parser.add_argument('-f', '--board-fen', metavar='<file>',
            required=True, type=str,
            help='first field of FEN describing board to search for')
    search_parser.add_argument('-x', '--chex-index', metavar='<dir>',
            required=True, type=str,
            help='chex index directory'
        )
    # Test various values!
    search_parser.add_argument('--search-k', metavar='<int>',
            required=False, type=int,
            default=-1,
            help='annoy search-k; default is results * n_trees'
        )
    search_parser.add_argument('--results', metavar='<int>',
            required=False, type=int,
            default=10,
            help='maximum number of returned game states'
        )
    args = parser.parse_args()
    if args.subparser_name == 'index':
        # print args.chex_index, args.id_label, args.first_indexed_move, args.n_trees, args.pgns
        index = ChexIndex(chex_index=args.chex_index, id_label=args.id_label,
                            first_indexed_move=args.first_indexed_move,
                            n_trees=args.n_trees)

        for pgn in args.pgns:
            game_count = 0
            with open(pgn) as pgn_stream:
                while True:
                    if index.add_game(chess.pgn.read_game(pgn_stream)):
                        break
                    game_count += 1
                    print 'Indexed {} games...\r'.format(game_count),
                    sys.stdout.flush()
        index.save()
        # TODO: clean up display of this
        print 'Indexed {} games.'.format(game_count)
    else:
        assert args.subparser_name == 'search'
        searcher = ChexSearch(args.chex_index,
                                results=args.results, search_k=args.search_k)
        # Pretty print results
        print '\t'.join(
                    ['rank', 'board FEN', 'similarity score', 'games',
                        'move numbers']
                )
        for (rank, (board, similarity, games)) in enumerate(searcher.search(
                    chess.Board(args.board_fen + ' w KQkq - 0 1')
                )):
            games = zip(*games)
            print '\t'.join([
                    str(rank + 1), board.board_fen(), str(similarity),
                    ','.join(games[0]), ','.join(map(str, games[1]))
                ])
        # Close may avoid shutdown exception for unknown reason
        searcher.close()
