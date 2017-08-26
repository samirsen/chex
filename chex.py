#!/usr/bin/env python
"""
chex

Indexes chess game states from one or more PGN files
(https://en.wikipedia.org/wiki/Portable_Game_Notation) with Spotify's annoy
(https://github.com/spotify/annoy) so the user can search for game states
similar to a game state they input as well as the games in which they're
found.
"""
import chess


_help_intro = """chex is a search engine for chess game states."""

def help_formatter(prog):
    """ So formatter_class's max_help_position can be changed. """
    return argparse.HelpFormatter(prog, max_help_position=40)

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
    search_parser.add_argument('-p', '--pgn', metavar='<file>',
            required=True, type=str,
            help='PGN describing game with state to search for')
    search_parser.add_argument('-m', '--move', metavar='<int>',
            required=True, type=int,
            help='move number from PGN corresponding to state to search for')
    args = parser.parse_args()
