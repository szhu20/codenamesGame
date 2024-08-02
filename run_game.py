import random
import time
import json
import enum
import os
import shutil
import sys

import colorama
import gensim.models.keyedvectors as word2vec
import numpy as np
from nltk.corpus import wordnet_ic

BOARD_SIZE = 5 # for board of size BOARD_SIZE x BOARD_SIZE
NUM_RED_WORDS = 9
NUM_BLUE_WORDS = 8
NUM_CIVILIAN_WORDS = BOARD_SIZE ** 2 - NUM_RED_WORDS - NUM_BLUE_WORDS - 1

class GameCondition(enum.Enum):
    """Enumeration that represents the different states of the game"""
    HIT_RED_AS_RED = 0
    CONTINUE_RED = 1
    HIT_BLUE_AS_BLUE = 2
    CONTINUE_BLUE = 3
    BLUE_WIN = 4
    RED_WIN = 5
    
RED_STATES = {GameCondition.HIT_RED_AS_RED, GameCondition.CONTINUE_RED}
BLUE_STATES = {GameCondition.HIT_BLUE_AS_BLUE, GameCondition.CONTINUE_BLUE}
CONTINUE_STATES = {GameCondition.HIT_RED_AS_RED, GameCondition.HIT_BLUE_AS_BLUE}
SWITCH_STATES = {GameCondition.CONTINUE_BLUE, GameCondition.CONTINUE_RED}
WIN_STATES = {GameCondition.RED_WIN, GameCondition.BLUE_WIN}

class Game:
    """Class that setups up game details and calls Guesser/Codemaster pair to play the game
    """

    def __init__(self, codemaster_red, guesser_red, codemaster_blue, guesser_blue,
                 seed="time", do_print=True, do_log=True, do_display=True, do_transcript=False,
                 game_name="default", log_name="default", transcript_name="default", cm_kwargs_r={},
                 g_kwargs_r={}, cm_kwargs_b={}, g_kwargs_b={}):
        """ Setup Game details

        Args:
            codemaster_red (:class:`Codemaster`):
                Red Codemaster (spymaster in Codenames' rules) class that provides a clue.
            guesser_red (:class:`Guesser`):
                Red Guesser (field operative in Codenames' rules) class that guesses based on clue.
            codemaster_blue (:class:`Codemaster`):
                Blue Codemaster (spymaster in Codenames' rules) class that provides a clue.
            guesser_blue (:class:`Guesser`):
                Blue Guesser (field operative in Codenames' rules) class that guesses based on clue.
            seed (int or str, optional): 
                Value used to init random, "time" for time.time(). 
                Defaults to "time".
            do_print (bool, optional): 
                Whether to keep on sys.stdout or turn off. 
                Defaults to True.
            do_log (bool, optional): 
                Whether to append to log file or not. 
                Defaults to True.
            do_display (bool, optional):
                Whether to display (print) the board.
                Defaults to True.
            game_name (str, optional): 
                game name used in log file. Defaults to "default".
            log_name (str, optional):
                log name used in log file name. Defaults to None.
            cm_kwargs_r (dict, optional): 
                kwargs passed to Red Codemaster.
            g_kwargs_r (dict, optional): 
                kwargs passed to Red Guesser.
            cm_kwargs_b (dict, optional): 
                kwargs passed to Blue Codemaster.
            g_kwargs_b (dict, optional): 
                kwargs passed to Blue Guesser.
        """

        self.game_start_time = time.time()
        colorama.init()

        self.do_print = do_print
        self.do_transcript = do_transcript
        if self.do_transcript:
            self._save_stdout = sys.stdout
            self.transcript = open(f"{transcript_name}_transcript.txt", 'a')
            sys.stdout = self.transcript
        elif not self.do_print:
            self._save_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        self.codemaster_red = codemaster_red(team="Red", **cm_kwargs_r)
        self.guesser_red = guesser_red(**g_kwargs_r)

        self.codemaster_blue = codemaster_blue(team="Blue", **cm_kwargs_b)
        self.guesser_blue = guesser_blue(**g_kwargs_b)

        print("#"*50)
        print(f"Red Codemaster: {codemaster_red.get_name()}")
        print(f"Blue Codemaster: {codemaster_blue.get_name()}")
        print(f"Red Guesser: {guesser_red.get_name()}")
        print(f"Blue Guesser: {guesser_blue.get_name()}")
        print("#"*50)
        self.cm_kwargs_r = cm_kwargs_r
        self.g_kwargs_r = g_kwargs_r
        self.cm_kwargs_b = cm_kwargs_b
        self.g_kwargs_b = g_kwargs_b
        self.do_log = do_log
        self.do_display = do_display
        self.game_name = game_name
        self.log_name = log_name

        # set seed so that board/keygrid can be reloaded later
        if seed == 'time':
            self.seed = time.time()
            random.seed(self.seed)
        else:
            self.seed = seed
            random.seed(int(seed))


        print(f"Red Codemaster: {codemaster_red.get_name()}")
        print(f"Red Guesser: {guesser_red.get_name()}")
        print(f"Blue Codemaster: {codemaster_blue.get_name()}")
        print(f"Blue Guesser: {guesser_blue.get_name()}")
        print("seed:", self.seed)

        # load board words
        with open("game_wordpool.txt", "r") as f:
            temp = f.read().splitlines()
            assert len(temp) == len(set(temp)), "game_wordpool.txt should not have duplicates"
            random.shuffle(temp)
            self.words_on_board = temp[:BOARD_SIZE ** 2]

        # set grid key for codemaster (spymaster)
        self.key_grid = ["Red"] * NUM_RED_WORDS + ["Blue"] * NUM_BLUE_WORDS + ["Civilian"] * NUM_CIVILIAN_WORDS + ["Assassin"]
        random.shuffle(self.key_grid)

    def __del__(self):
        """reset stdout if using the do_print==False option"""
        if not self.do_print:
            sys.stdout.close()
            sys.stdout = self._save_stdout

    @staticmethod
    def load_glove_vecs(glove_file_path):
        """Load stanford nlp glove vectors
        Original source that matches the function: https://nlp.stanford.edu/data/glove.6B.zip
        """
        if glove_file_path == "default":
            glove_file_path = "glove/glove.6B.100d.txt"
        with open(glove_file_path, encoding="utf-8") as infile:
            glove_vecs = {}
            for line in infile:
                line = line.rstrip().split(' ')
                glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])
            return glove_vecs

    @staticmethod
    def load_wordnet(wordnet_file):
        """Function that loads wordnet from nltk.corpus"""
        if wordnet_file == "default":
            wordnet_file = "ic-brown.dat"
        return wordnet_ic.ic(wordnet_file)

    @staticmethod
    def load_w2v(w2v_file_path):
        """Function to initalize gensim w2v object from Google News w2v Vectors
        Vectors Source: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        """
        if w2v_file_path == "default":
            w2v_file_path = "GoogleNews-vectors-negative300.bin" 
        return word2vec.KeyedVectors.load_word2vec_format(w2v_file_path, binary=True, unicode_errors='ignore')

    def _display_board_codemaster(self):
        """prints out board with color-paired words, only for codemaster, color && stylistic"""
        print(str.center("___________________________BOARD___________________________\n", 60))
        counter = 0
        for i in range(len(self.words_on_board)):
            if counter >= 1 and i % BOARD_SIZE == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RED + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.BLUE + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.MAGENTA + self.words_on_board[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 60))
        print("\n")

    def _display_board(self):
        """prints the list of words in a board like fashion"""
        print(colorama.Style.RESET_ALL)
        print(str.center("___________________________BOARD___________________________", 60))
        for i in range(len(self.words_on_board)):
            if i % BOARD_SIZE == 0:
                print("\n")
            print(str.center(self.words_on_board[i], 10), " ", end='')

        print(str.center("\n___________________________________________________________", 60))
        print("\n")

    def _display_key_grid(self):
        """ Print the key grid to stdout  """
        print("\n")
        print(str.center(colorama.Fore.RESET +
                         "____________________________KEY____________________________\n", 55))
        counter = 0
        for i in range(len(self.key_grid)):
            if counter >= 1 and i % BOARD_SIZE == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RED + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.BLUE + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.MAGENTA + self.key_grid[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 55))
        print("\n")

    def _print_word_lists(self):
        """Print the list of good words and list of bad words"""
        good_words = []
        bad_words = []
        for word, key in zip(self.words_on_board, self.key_grid):
            if word[0] != '*':
                if key == 'Red':
                    good_words.append(word)
                else:
                    bad_words.append(word)

        print(f"good words: {good_words}")
        print(f"bad words: {bad_words}\n")

    def get_words_on_board(self):
        """Return the list of words that represent the board state"""
        return self.words_on_board

    def get_key_grid(self):
        """Return the codemaster's key"""
        return self.key_grid

    def _accept_guess(self, guess_index, game_condition):
        """Function that takes in an int index called guess to compare with the key grid
        CodeMaster will always win with Red and lose if Blue =/= 8 or Assassin == 1
        """
        is_red = game_condition in RED_STATES
        if self.key_grid[guess_index] == "Red":
            self.words_on_board[guess_index] = "*Red*"
            if self.words_on_board.count("*Red*") >= NUM_RED_WORDS:
                return GameCondition.RED_WIN
            return GameCondition.HIT_RED_AS_RED if is_red else GameCondition.CONTINUE_RED

        elif self.key_grid[guess_index] == "Blue":
            self.words_on_board[guess_index] = "*Blue*"
            if self.words_on_board.count("*Blue*") >= NUM_BLUE_WORDS:
                return GameCondition.BLUE_WIN
            else:
                return GameCondition.CONTINUE_BLUE if is_red else GameCondition.HIT_BLUE_AS_BLUE

        elif self.key_grid[guess_index] == "Assassin":
            self.words_on_board[guess_index] = "*Assassin*"
            return GameCondition.BLUE_WIN if is_red else GameCondition.RED_WIN

        else:
            self.words_on_board[guess_index] = "*Civilian*"
            return GameCondition.CONTINUE_BLUE if is_red else GameCondition.CONTINUE_RED

    def write_results(self, num_of_turns, game_condition):
        """Logging function
        writes in both the original and a more detailed new style
        """
        red_result = 0
        blue_result = 0
        civ_result = 0
        assa_result = 0

        for i in range(len(self.words_on_board)):
            if self.words_on_board[i] == "*Red*":
                red_result += 1
            elif self.words_on_board[i] == "*Blue*":
                blue_result += 1
            elif self.words_on_board[i] == "*Civilian*":
                civ_result += 1
            elif self.words_on_board[i] == "*Assassin*":
                assa_result += 1
        # total = red_result + blue_result + civ_result + assa_result

        if not os.path.exists("results"):
            os.mkdir("results")
        with open("results/bot_results.txt", "a") as f:
            f.write(
                f'TOTAL:{num_of_turns} B:{blue_result} C:{civ_result} A:{assa_result}'
                f' R:{red_result} CM_R:{type(self.codemaster_red).get_name()} '
                f'GUESSER_R:{type(self.guesser_red).get_name()} CM_B:{type(self.codemaster_blue).get_name()} '
                f'GUESSER_B:{type(self.guesser_blue).get_name()} SEED:{self.seed}\n'
            )
        
        if self.log_name is not None:
            with open(f"{self.log_name}.txt", "a") as f:
                results = {"game_name": self.game_name,
                        "total_turns": num_of_turns,
                        "winner": "red" if game_condition == GameCondition.RED_WIN else "blue",
                        "R": red_result, "B": blue_result, "C": civ_result, "A": assa_result,
                        "codemaster_red": type(self.codemaster_red).get_name(),
                        "guesser_red": type(self.guesser_red).get_name(),
                        "codemaster_blue": type(self.codemaster_blue).get_name(),
                        "guesser_blue": type(self.guesser_blue).get_name(),
                        "seed": self.seed,
                        "time_s": (self.game_end_time - self.game_start_time),
                        "cm_kwargs_r": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                        for k, v in self.cm_kwargs_r.items()},
                        "g_kwargs_r": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                        for k, v in self.g_kwargs_r.items()},
                        "cm_kwargs_b": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                        for k, v in self.cm_kwargs_b.items()},
                        "g_kwargs_b": {k: v if isinstance(v, float) or isinstance(v, int) or isinstance(v, str) else None
                                        for k, v in self.g_kwargs_b.items()},
                        }
                f.write(json.dumps(results))
                f.write('\n')

    @staticmethod
    def clear_results():
        """Delete results folder"""
        if os.path.exists("results") and os.path.isdir("results"):
            shutil.rmtree("results")

    def run(self):
        """Function that runs the codenames game between codemaster and guesser"""
        game_condition = GameCondition.CONTINUE_RED
        game_counter = 0
        while game_condition not in WIN_STATES:
            # board setup and display
            print('\n' * 2)
            words_in_play = self.get_words_on_board()
            current_key_grid = self.get_key_grid()

            codemaster = None
            guesser = None
            if game_condition in RED_STATES:
                codemaster = self.codemaster_red
                guesser = self.guesser_red
            else:
                codemaster = self.codemaster_blue
                guesser = self.guesser_blue
            codemaster.set_game_state(words_in_play, current_key_grid)
            if self.do_display:
                self._display_key_grid()
                self._display_board_codemaster()
            # self._print_word_lists()

            # codemaster gives clue & number here
            clue, clue_num = codemaster.get_clue()
            game_counter += 1
            keep_guessing = True
            guess_num = 0
            clue_num = int(clue_num)

            print('\n' * 2)
            guesser.set_clue(clue, clue_num)

            if game_condition in RED_STATES:
                game_condition = GameCondition.HIT_RED_AS_RED
            else:
                game_condition = GameCondition.HIT_BLUE_AS_BLUE
            while guess_num <= clue_num and keep_guessing and game_condition in CONTINUE_STATES:
                guesser.set_board(words_in_play)
                guess_answer = guesser.get_answer()

                # if no comparisons were made/found than retry input from codemaster
                if guess_answer is None or guess_answer == "no comparisons":
                    break
                guess_answer_index = words_in_play.index(guess_answer.upper().strip())
                game_condition = self._accept_guess(guess_answer_index, game_condition)

                if game_condition in CONTINUE_STATES:
                    print('\n' * 2)
                    if self.do_display:
                        self._display_board_codemaster()
                    guess_num += 1
                    print("Keep Guessing? the clue is ", clue, clue_num)
                    keep_guessing = guesser.keep_guessing()

                # if guesser selected a civilian or an opposite-paired word
                elif game_condition in SWITCH_STATES:
                    break

                elif game_condition in WIN_STATES:
                    self.game_end_time = time.time()
                    if self.do_display:
                        self._display_board_codemaster()
                    if self.do_log:
                        self.write_results(game_counter, game_condition)
                    print(f'{"Red" if game_condition == GameCondition.RED_WIN else "Blue"} Won')
                    print("Game Counter:", game_counter)
                    return game_condition == GameCondition.RED_WIN

                # elif game_condition == GameCondition.WIN:
                #     self.game_end_time = time.time()
                #     if self.do_display:
                #         self._display_board_codemaster()
                #     if self.do_log:
                #         self.write_results(game_counter)
                #     print("You Won")
                #     print("Game Counter:", game_counter)

            if guess_num == clue_num + 1 or not keep_guessing:
                game_condition = GameCondition.CONTINUE_BLUE if game_condition in RED_STATES else GameCondition.CONTINUE_RED
