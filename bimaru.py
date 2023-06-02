# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 85:
# 103403 Guilherme Henriques
# 104126 Fábio Neto

from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
)

# Board settings
BOARD_SIZE = 10      # number of columns/rows
BOATS = [1, 2, 3, 4] # boats to add in inverse order (big to small).

# How much a hint, a cell and a boat count as an heuristic
hhint = 1
hcell = 0
hboat = 1

# Non-pieces
EMPTY, WATER = (None, 'w')

# Center Pieces
CIRCLE, MID = ('c', 'm')

# Directional Pieces
LEFT, RIGHT, TOP, BOT = ('l', 'r', 't', 'b')

def new_grid():
    """Creates a new grid of size BOARD_SIZE."""
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, row_values, col_values, h, hints = new_grid(), grid = new_grid(), boats = BOATS):
        self.row_values = row_values
        self.col_values = col_values
        self.h = h
        self.hints = hints
        self.grid = grid
        self.boats = boats

    def __clone__(self):
        """Creates a complete clone from this board.
        Hints are not copied as they can be shared among boards."""
        row_values = self.row_values.copy()
        col_values = self.col_values.copy()
        grid = [self.grid[r].copy() for r in range(BOARD_SIZE)]
        boats = self.boats.copy()

        return Board(row_values, col_values, self.h, self.hints, grid, boats)
    
    def is_goal(self):
        """Returns True if this board reached a successful (goal) state"""
        if sum(self.row_values) or sum(self.boats):
            return False

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                hint = self.hints[r][c]

                if (hint is not EMPTY) and (self.grid[r][c] != hint.lower()):
                    return False

        return True

    def is_pos_valid(self, r: int, c: int):
        """Returns True if the given position is valid for this board."""
        return (0 <= r < BOARD_SIZE) and (0 <= c < BOARD_SIZE)

    def get_value(self, r: int, c: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.grid[r][c] if self.is_pos_valid(r, c) else EMPTY

    def set_value(self, r: int, c: int, value: str):
        """Define um novo valor para a posição dada."""
        if not self.is_pos_valid(r, c):
            return
        
        if (self.grid[r][c] is EMPTY):
            self.h -= hcell

            if (value.upper() == self.hints[r][c]):
                self.h -= hhint

        self.grid[r][c] = value

    def fill_row(self, r: int):
        """"Fills the given row with water."""
        for c in range(BOARD_SIZE):
            if (self.grid[r][c] is EMPTY):
                self.set_value(r, c, WATER)

    def fill_column(self, c: int):
        """Fills the given column with water."""
        for r in range(BOARD_SIZE):
            if (self.grid[r][c] is EMPTY):
                self.set_value(r, c, WATER)
    
    def fill_zeros(self):
        """Fills all complete rows and columns (whose values are 0) with water."""
        for i in range(BOARD_SIZE):
            if not self.row_values[i]:
                self.fill_row(i)
            
            if not self.col_values[i]:
                self.fill_column(i)
    
    def fits_vboat(self, r: int, c: int, size: int):
        """Returns True if it possible to insert a vertical boat in the given position."""
        if (self.col_values[c] < size) or (r + size > BOARD_SIZE):
            return False

        for x in range(r, r+size):
            if (self.grid[x][c] is not EMPTY):
                return False
        
        return True
    
    def fits_hboat(self, r: int, c: int, size: int):
        """Returns True if it is possible to insert a horizontal boat in the given position."""
        if (self.row_values[r] < size) or (c + size > BOARD_SIZE):
            return False

        for x in range(c, c+size):
            if (self.grid[r][x] is not EMPTY):
                return False
            
        return True

    
    def decrement_col_value(self, c: int, amnt: int):
        """Decrement the column value, and fills with water if its value reached 0."""
        if (c < 0) or (c >= BOARD_SIZE):
            return
        
        self.col_values[c] -= amnt

        if not self.col_values[c]:
            self.fill_column(c)
    
    def decrement_row_value(self, r: int, amnt: int):
        """Decrement the row value, and fills with water if its value reached 0."""
        if (r < 0) or (r >= BOARD_SIZE):
            return
        
        self.row_values[r] -= amnt

        if not self.row_values[r]:
            self.fill_row(r)
    
    def generate_actions(self):
        """Generate all actions this board is capable of executing.

        An action consists of a tuple representing the possibility of inserting a boat,
        where the first two values are its position, the third is its size and the fourth
        is a boolean indicating its direction (True = vertical, False = horizontal).
        """
        size = len(self.boats)

        for i in range(size):
            if self.boats[i]:
                size -= i
                break
        else:
            return ()
        
        result = []

        if (size == 1):
            for r in range(BOARD_SIZE):
                if not self.row_values[r]:
                    continue

                for c in range(BOARD_SIZE):
                    if (self.grid[r][c] is EMPTY):
                        result.append((r, c, size))

            return () if (len(result) < self.boats[-size]) else result
        
        for i in range(BOARD_SIZE):
            for j in range(11 - size):
                if self.fits_vboat(j, i, size):
                    result.append((j, i, size, True))

                if self.fits_hboat(i, j, size):
                    result.append((i, j, size, False))

        return result

    
    def execute_action(self, r: int, c: int, size: int, vertical = False):
        """Executes an action (inserts a boat), ideally generated by generate_actions."""
        clone = self.__clone__()

        clone.h -= hboat
        clone.boats[-size] -= 1

        clone.set_value(r - 1, c - 1, WATER)
        clone.set_value(r - 1, c, WATER)
        clone.set_value(r, c - 1, WATER)

        if (size == 1):
            clone.set_value(r, c, CIRCLE)
            clone.set_value(r + 1, c, WATER)
            clone.set_value(r, c + 1, WATER)
            clone.set_value(r - 1, c + 1, WATER)
            clone.set_value(r + 1, c - 1, WATER)
            clone.set_value(r + 1, c + 1, WATER)
            clone.decrement_row_value(r, 1)
            clone.decrement_col_value(c, 1)
            return clone

        if (vertical):
            clone.decrement_col_value(c, size)
            clone.decrement_row_value(r, 1)
            clone.set_value(r, c, TOP)
            clone.set_value(r - 1, c + 1, WATER)
            clone.set_value(r, c + 1, WATER)

            if (size > 2):
                r += 1
                clone.set_value(r, c, MID)
                clone.set_value(r, c - 1, WATER)
                clone.set_value(r, c + 1, WATER)
                clone.decrement_row_value(r, 1)

                if (size > 3):
                    r += 1
                    clone.set_value(r, c, MID)
                    clone.set_value(r, c + 1, WATER)
                    clone.set_value(r, c - 1, WATER)
                    clone.decrement_row_value(r, 1)
            
            r += 1
            clone.set_value(r, c, BOT)
            clone.set_value(r, c - 1, WATER)
            clone.set_value(r, c + 1, WATER)
            clone.set_value(r + 1, c, WATER)
            clone.set_value(r + 1, c - 1, WATER)
            clone.set_value(r + 1, c + 1, WATER)
            clone.decrement_row_value(r, 1)
            return clone

        clone.decrement_col_value(c, 1)
        clone.decrement_row_value(r, size)
        clone.set_value(r, c, LEFT)
        clone.set_value(r + 1, c - 1, WATER)
        clone.set_value(r + 1, c, WATER)

        if (size > 2):
            c += 1
            clone.set_value(r, c, MID)
            clone.set_value(r - 1, c, WATER)
            clone.set_value(r + 1, c, WATER)
            clone.decrement_col_value(c, 1)

            if (size > 3):
                c += 1
                clone.set_value(r, c, MID)
                clone.set_value(r - 1, c, WATER)
                clone.set_value(r + 1, c, WATER)
                clone.decrement_col_value(c, 1)
        
        c += 1
        clone.set_value(r, c, RIGHT)
        clone.set_value(r, c + 1, WATER)
        clone.set_value(r - 1, c, WATER)
        clone.set_value(r + 1, c, WATER)
        clone.set_value(r - 1, c + 1, WATER)
        clone.set_value(r + 1, c + 1, WATER)
        clone.decrement_col_value(c, 1)
        return clone
    
    def add_hint(self, r: int, c: int, value: str):
        """Adds a hint."""
        if not self.is_pos_valid(r, c):
            return

        self.hints[r][c] = value

        if (value == 'W'):
            self.set_value(r, c, WATER)
            return

        self.set_value(r - 1, c - 1, WATER)
        self.set_value(r - 1, c + 1, WATER)
        self.set_value(r + 1, c - 1, WATER)
        self.set_value(r + 1, c + 1, WATER)

        if (value == 'M'):
            return

        if (value == 'C'):
            # .  .  . (row - 1)
            # .  c  . (row)
            # .  .  . (row + 1)
            self.set_value(r, c, CIRCLE)
            self.set_value(r - 1, c, WATER)
            self.set_value(r + 1, c, WATER)
            self.set_value(r, c - 1, WATER)
            self.set_value(r, c + 1, WATER)
            self.decrement_col_value(c, 1)
            self.decrement_row_value(r, 1)
            self.boats[-1] -= 1
        elif (value == 'L'):
            # . . . (row - 1)
            # . l ? (row)
            # . . . (row + 1)
            self.set_value(r - 1, c, WATER)
            self.set_value(r + 1, c, WATER)
            self.set_value(r, c - 1, WATER)
        elif (value == 'R'):
            # . . . (row - 1)
            # ? r . (row)
            # . . . (row + 1)
            self.set_value(r - 1, c, WATER)
            self.set_value(r + 1, c, WATER)
            self.set_value(r, c + 1, WATER)
        elif (value == 'T'):
            # . . . (row - 1)
            # . t . (row)
            # . ? . (row + 1)
            self.set_value(r - 1, c, WATER)
            self.set_value(r, c - 1, WATER)
            self.set_value(r, c + 1, WATER)
        elif (value == 'B'):
            # . ? . (row - 1)
            # . b . (row)
            # . . . (row + 1)
            self.set_value(r + 1, c, WATER)
            self.set_value(r, c - 1, WATER)
            self.set_value(r, c + 1, WATER)


    def print(self):
        line = ''

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                value = self.hints[r][c]

                if (value is EMPTY):
                    value = self.grid[r][c]

                    if (value is WATER):
                        value = '.'
                    elif (value is EMPTY):
                        value = '?'

                line += value
            
            line += '\n'

        print(line, end="")

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        row_values = [int(i) for i in stdin.readline().split()[1:]]
        col_values = [int(i) for i in stdin.readline().split()[1:]]
        hints = int(stdin.readline())
        board = Board(row_values, col_values, hints * hhint + hcell * 100 + hboat * 10)

        for _ in range(hints):
            line = stdin.readline().split()[1:]  # Ignores 'HINT\t'

            board.add_hint(int(line[0]), int(line[1]), line[2])
        
        board.fill_zeros()

        return board


class BimaruState:
    state_id = 0

    def __init__(self, board: Board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id > other.id


class Bimaru(Problem):

    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(BimaruState(board))

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        return state.board.generate_actions()

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        return BimaruState(state.board.execute_action(*action))

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        return state.board.is_goal()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return node.state.board.h

    # TODO: outros metodos da classe


def example2():
    board = Board.parse_instance()

    problem = Bimaru(board)
    initial_state = BimaruState(board)

    print(initial_state.board.get_value(3, 3))

    result_state = problem.result(initial_state, (3, 3, WATER))

    print(result_state.board.get_value(3, 3))


def test():
    problem = Bimaru(Board.parse_instance())
    goal = depth_first_tree_search(problem)

    if goal is None:
        print('Error: goal is none!')
        return

    goal.state.board.print()


if __name__ == "__main__":
    test()

    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
