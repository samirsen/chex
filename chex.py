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
import random
import atexit
import shutil
import copy
import tempfile
import logging
from annoy import AnnoyIndex
from sqlitedict import SqliteDict

_help_intro = """chex is a search engine for chess game states."""

def help_formatter(prog):
    """ So formatter_class's max_help_position can be changed. """
    return argparse.HelpFormatter(prog, max_help_position=40)

# For bitboard conversion
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

_bitboard_length = 768

def board_to_bitboard(board):
    """ Converts chess module's board to bitboard game state representation.

        node: game object of type chess.pgn.Game

        Return value: binary vector of length _bitboard_length as Python list
    """
    bitboard = [0 for _ in xrange(_bitboard_length)]
    for i in xrange(64):
        try:
            bitboard[i*12 + _offsets[board.piece_at(i).symbol()]] = 1
        except AttributeError:
            pass
    return bitboard

def bitboard_to_board(bitboard):
    """ Converts bitboard to board.

        TODO: unit test.

        bitboard: iterable of _bitboard_length 1s and 0s

        Return value: chess.Board representation of bitboard
    """
    fen = [[] for _ in xrange(8)]
    for i in xrange(8):
        streak = 0
        for j in xrange(8):
            segment = 12*(8*i + j)
            piece = None
            for offset in xrange(12):
                if bitboard[segment + offset]:
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

def bitboard_to_key(bitboard):
    """ Converts bitboard to ASCII representation used as key in SQL database.

        bitboard: bitboard representation of chess board

        Return value: ASCII representation of bitboard
    """
    to_unhexlify = '%x' % int(''.join(map(str, map(int, bitboard))), 2)
    try:
        return binascii.unhexlify(to_unhexlify)
    except TypeError:
        return binascii.unhexlify('0' + to_unhexlify)

def key_to_bitboard(key):
    """ Converts ASCII representation of board to bitboard.

        key: ASCII representation of bitboard

        Return value: bitboard (binary list)
    """
    unpadded = [
            int(digit) for digit in bin(int(binascii.hexlify(key), 16))[2:]]
    return [0 for _ in xrange(_bitboard_length - len(unpadded))] + unpadded


def invert_board(board):
    """ Computes bitboard of given position but with inverted colors. """
    inversevector = [0 for _ in xrange(_bitboard_length)]
    for i in xrange(64):
        try:
            inversevector[i * 12
                    + _reverse_colors_offsets[board.piece_at(i).symbol()]] = 1
        except AttributeError:
            pass
    return inversevector

def flip_board(board):
    """ Computes bitboard of the mirror image of a given position. """
    flipvector = [0 for _ in xrange(_bitboard_length)]
    for i in range(8):
        for j in range(8):
            try:
                flipvector[12*(8*i + 7 - j)
                    + _offsets[board.piece_at(8*i + j).symbol()]] = 1
            except AttributeError:
                pass

    return flipvector

def reverse_and_flip(board):
    """ Computes bitboard after flipping position and reversing colors.

        board: object of type chess.Board

        Return value: flipped bitboard
        """
    reversevector = [0 for _ in xrange(_bitboard_length)]
    for i in range(8):
        for j in range(8):
            try:
                reversevector[12*(8*i + 7 - j)
                                + _reverse_colors_offsets[
                                        board.piece_at(8*i + j).symbol()]] = 1
            except AttributeError:
                pass

    return reversevector


class ChexIndex(AnnoyIndex):
    """ Manages game states from Annoy Index and SQL database. """

    def __init__(self, chex_index, id_label='FICSGamesDBGameNo',
                    first_indexed_move=10, n_trees=200, seed=1,
                    scratch=None, learning_rate=1, min_iterations=100,
                    max_iterations=5000000, difference=.1):
        """ Number of dimensions is always 8 x 8 x 12; there are 6 black piece
        types, six white piece types, and the board is 8 x 8."""
        super(ChexIndex, self).__init__(_bitboard_length, metric='angular')
        self.id_label = id_label
        self.first_indexed_move = first_indexed_move
        self.chex_index = chex_index
        try:
            os.makedirs(self.chex_index)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # Create temporary directory
        if scratch is not None:
            try:
                os.makedirs(self.scratch)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        self.scratch = tempfile.mkdtemp(dir=scratch)
        # Schedule temporary directory for deletion
        atexit.register(shutil.rmtree, self.scratch, ignore_errors=True)
        self.chex_sql = SqliteDict(
                            os.path.join(self.chex_index, 'sqlite.idx'))
        self.game_sql = SqliteDict(
                            os.path.join(self.scratch, 'temp.idx')
                        )
        self.game_number = 0
        self.n_trees = n_trees
        # For reproducibly randomly drawing boards
        self.seed = seed
        self.learning_rate = learning_rate
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.difference = difference
        self.weights = [0 for _ in xrange(_bitboard_length)]

    def add_game(self, node):
        """ Adds game parsed by chess library to chex SQL database.

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

            bitboard = board_to_bitboard(node.board())
            inversevector = invert_board(node.board())
            flipvector = flip_board(node.board())
            reversevector = reverse_and_flip(node.board())

            # Store as ASCII; use minimum of strategically equivalent boards
            # See https://github.com/samirsen/chex/issues/1 for details
            key = min(map(bitboard_to_key,
                        [bitboard, inversevector, flipvector, reversevector]))
            if key in self.chex_index:
                self.chex_sql[key] = self.chex_sql[key] + [
                                                        (game_id, move_number)
                                                    ]
            else:
                self.chex_sql[key] = [(game_id, move_number)]
            if self.game_number in self.game_sql:
                self.game_sql[self.game_number].append(key)
            else:
                self.game_sql[self.game_number] = [key]
            if node.is_end(): break
            node = node.variations[0]
        self.game_number += 1
        return 0

    def _mahalanobis_loss(self,
                reference_bitboard, plus_bitboard, minus_bitboard):
        """ Computes value of loss function for finding Mahalanobis metric.

            reference_bitboard, plus_bitboard, minus_bitboard: explained
                in algo

            Return value: value of loss function 
        """
        return max(0.,
                1.  + sum([minus_bitboard[i]
                            * reference_bitboard[i] * self.weights[i]
                            for i in xrange(_bitboard_length)])
                   - sum([plus_bitboard[i]
                            * reference_bitboard[i] * self.weights[i]
                            for i in xrange(_bitboard_length)]))

    def _mahalanobis(self):
        """ Computes sparse Mahalanobis metric using algorithm from paper.

            The reference is SOML: Sparse online metric learning with
                application to image retrieval by Gao et al. We implement
                their algorithm 1: SOML-TG (sparse online metric learning via
                truncated gradient). We set lambda = 0 and use no
                sparsity-promoting regularization term.

            Return value: diagonal of Mahalanobis metric
        """
        # Finalize game SQL database for querying
        self.game_sql.commit()
        # For reproducible random draws from database
        random.seed(self.seed)
        last_weights = [0 for _ in xrange(_bitboard_length)]
        iteration, critical_iteration = 0, self.min_iterations
        while True:
            # Draw game
            game_index = random.randint(0, self.game_number)
            # Check that the sampled boards are shuffled
            # Is the Python algo reservoir sampling? If so yes.
            [reference_bitboard, plus_bitboard, minus_bitboard] = map(
                                key_to_bitboard,
                                random.sample(
                                    list(enumerate(
                                            self.game_sql[str(game_index)])),
                                  3)
                            )
            if abs(minus_bitboard[0] - reference_bitboard[0]) < abs(
                plus_bitboard[0] - reference_bitboard[0]):
                minus_bitboard, plus_bitboard = plus_bitboard, minus_bitboard
            if self._mahalanobis_loss(reference_bitboard[1],
                                        plus_bitboard[1], minus_bitboard[1],
                                        self.weights) > 0:
                v = [self. weights[i] - self.learning_rate
                        * reference_bitboard[1][i]
                        * (plus_bitboard[1][i] - minus_bitboard[1][i])
                        for i in xrange(_bitboard_length)]
                self.weights = [max(0, v[j]) if v[j] >=0 else min(0, v[j])
                                for j in xrange(_bitboard_length)]
            iteration += 1
            if iteration >= critical_iteration:
                if sqrt(sum([(last_weights[i] - self.weights[i])**2
                                for i in xrange(_bitboard_length)])) <= (
                        self.difference):
                    # Must sqrt so angular distance in annoy works
                    self.weights = [sqrt(weight) for weight in self.weights]
                    break
                last_weights = copy.copy(self.weights)
                critical_iteration *= 2
            if iteration >= self.max_iterations:
                # Must sqrt so angular distance in annoy works
                self.weights = [sqrt(weight) for weight in self.weights]
                break

    def _annoy_index(self):
        """ Adds all boards from chex SQL database to Annoy index

            No return value.
        """
        for i, key in enumerate(self.chex_sql):
            bitboard = key_to_bitboard(key)
            self.add_item(i, [self.weights[j] * bitboard[j]
                                for j in xrange(_bitboard_length)])

    def save(self):
        # Compute Mahalanobis matrix
        self._mahalanobis()
        # Create annoy index
        self._annoy_index()
        self.build(self.n_trees)
        # Save all index files
        super(ChexIndex, self).save(
                os.path.join(self.chex_index, 'annoy.idx')
            )
        self.chex_sql.commit()
        self.chex_sql.close()
        self.game_sql.close()
        # Clean up
        shutil.rmtree(self.scratch, ignore_errors=True)

class ChexSearch(object):
    """ Searches Chex index for game states and associated games. """

    #TODO: Combine results of board transforms with binary search algo.

    def __init__(self, chex_index, results=10, search_k=40):
        self.chex_index = chex_index
        self.results = results
        self.search_k = search_k
        self.annoy_index = AnnoyIndex(_bitboard_length, metric='angular')
        self.annoy_index.load(os.path.join(self.chex_index, 'annoy.idx'))
        self.chex_sql = SqliteDict(
                            os.path.join(self.chex_index, 'sqlite.idx'))

    def search(self, board):
        """ Searches for board.

            board: game object of type chess.Board

            Return value: [
                (board, similarity score, [(game_id, move number), ...]), ...]
        """

        symmetrical_boards = [board_to_bitboard(board),
                                invert_board(board),
                                flip_board(board),
                                reverse_and_flip(board)]
        results = []
        for bitboard in symmetrical_boards:
            for annoy_id, similarity in zip(
                                *self.annoy_index.get_nns_by_vector(
                                        bitboard, self.results,
                                        include_distances=True
                            )):
                # Recompute ASCII key
                bitboard = self.annoy_index.get_item_vector(annoy_id)
                to_unhexlify = '%x' % int(''.join(
                                            map(str, map(int, bitboard))), 2)
                try:
                    key = binascii.unhexlify(to_unhexlify)
                except TypeError:
                    key = binascii.unhexlify('0' + to_unhexlify)
                results.append((bitboard_to_board(bitboard), similarity,
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
    index_parser.add_argument('--scratch', metavar='<dir>', type=str,
            required=False,
            default=None,
            help=('where to store temporary files; default is securely '
                  'created directory in $TMPDIR or similar'))
    index_parser.add_argument('--learning_rate', metavar='<dec>', type=float,
            required=False,
            default=1,
            help='learning rate for Mahalanobis metric')
    index_parser.add_argument('--min-iterations', metavar='<int>', type=int,
            required=False,
            default=100,
            help='minimum number of iterations for learning Mahalanobis metric'
        )
    index_parser.add_argument('--max-iterations', metavar='<int>', type=int,
            required=False,
            default=100,
            help='maximum number of iterations for learning Mahalanobis metric'
        )
    index_parser.add_argument('--difference', metavar='<dec>', type=float,
            required=False,
            default=.1,
            help=('maximum Euclidean distance between Mahalanobis matrices '
                  'for deciding convergence')
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
    parser.add_argument('--verbose', action='store_const', const=True,
            default=False,
            help='be talkative'
        )
    args = parser.parse_args()
    # Configure this a little later
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-10s %(message)s',
                        datefmt='%m-%d-%Y %H:%M:%S')
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    if args.subparser_name == 'index':
        index = ChexIndex(args.chex_index, id_label=args.id_label,
                            first_indexed_move=args.first_indexed_move,
                            n_trees=args.n_trees, scratch=args.scratch,
                            learning_rate=args.learning_rate,
                            min_iterations=args.min_iterations,
                            max_iterations=args.max_iterations,
                            difference=args.difference)

        for pgn in args.pgns:
            game_count = 0
            with open(pgn) as pgn_stream:
                while True:
                    if index.add_game(chess.pgn.read_game(pgn_stream)):
                        break
                    game_count += 1
                    print 'Read {} games...\r'.format(game_count),
                    sys.stdout.flush()
        index.save()
        # TODO: clean up display of this
        print 'Read {} games.'.format(game_count)
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
