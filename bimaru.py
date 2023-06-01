# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 85:
# 103403 Guilherme Henriques
# 104126 Fábio Neto

import numpy as np

from sys import stdin
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

iterator10 = range(10)
iterator4 = range(4)

def create_grid():
    return [[None for _ in iterator10] for _ in iterator10]

def copy_grid(grid):
    return [grid[r].copy() for r in iterator10]


class Board:
    """Representação interna de um tabuleiro de Bimaru."""

    def __init__(self, row_values, col_values, hints = create_grid(), grid = create_grid(), boats = [1, 2, 3, 4]):
        self.row_values = row_values
        self.col_values = col_values
        self.hints = hints
        self.grid = grid
        self.boats = boats

    def clone(self):
        row_values = self.row_values.copy()
        col_values = self.col_values.copy()
        hints = copy_grid(self.hints)
        grid = copy_grid(self.grid)
        boats = self.boats.copy()

        return Board(row_values, col_values, hints, grid, boats)

    def is_pos_valid(self, r: int, c: int):
        return (0 <= r < 10) and (0 <= c < 10)

    def get_hint(self, r: int, c: int):
        return self.hints[r][c] if self.is_pos_valid(r, c) else None

    def get_value(self, r: int, c: int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.grid[r][c] if self.is_pos_valid(r, c) else None

    def set_value(self, r: int, c: int, value: str):
        if not self.is_pos_valid(r, c):
            return

        self.grid[r][c] = value

    def fill_row(self, r: int):
        for c in iterator10:
            if (self.grid[r][c] is None):
                self.set_value(r, c, 'w')

    def fill_column(self, c: int):
        for r in iterator10:
            if (self.grid[r][c] is None):
                self.set_value(r, c, 'w')
    
    def fill_zeros(self):
        for i in iterator10:
            if (self.row_values[i] == 0):
                self.fill_row(i)
            
            if (self.col_values[i] == 0):
                self.fill_column(i)

    def adjacent_vertical_values(self, r: int, c: int) -> tuple:
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""

        above = None if (r == 0) else self.grid[r - 1][c]
        below = None if (r == 9) else self.grid[r + 1][c]

        return above, below

    def adjacent_horizontal_values(self, r: int, c: int) -> tuple:
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""

        left = None if (c == 0) else self.grid[r][c - 1]
        right = None if (c == 9) else self.grid[r][c + 1]

        return left, right
    
    def fits_cboat(self, r: int, c: int):
        return (self.grid[r][c] is None)
    
    def fits_vboat(self, r: int, c: int, size: int):
        if (self.col_values[c] < size) or (r + size > 10):
            return False

        for x in range(r, r+size):
            if (self.grid[x][c] is not None):
                return False
        
        return True
    
    def fits_hboat(self, r: int, c: int, size: int):
        if (self.row_values[r] < size) or (c + size > 10):
            return False

        for x in range(c, c+size):
            if (self.grid[r][x] is not None):
                return False
            
        return True
    
    def decrement_col_value(self, c: int, amnt: int):
        if (c < 0) or (c > 9):
            return
        
        self.col_values[c] -= amnt

        if (self.col_values[c] == 0):
            self.fill_column(c)
    
    def decrement_row_value(self, r: int, amnt: int):
        if (r < 0) or (r > 9):
            return
        
        self.row_values[r] -= amnt

        if (self.row_values[r] == 0):
            self.fill_row(r)
    
    def insert_cboat(self, r: int, c: int):
        self.boats[-1] -= 1
        self.set_value(r, c, 'c')
        self.set_value(r - 1, c, 'w')
        self.set_value(r + 1, c, 'w')
        self.set_value(r, c - 1, 'w')
        self.set_value(r, c + 1, 'w')
        self.set_value(r - 1, c - 1, 'w')
        self.set_value(r - 1, c + 1, 'w')
        self.set_value(r + 1, c - 1, 'w')
        self.set_value(r + 1, c + 1, 'w')
        self.decrement_row_value(r, 1)
        self.decrement_col_value(c, 1)
    
    def insert_vboat(self, r: int, c: int, size: int):
        self.boats[-size] -= 1
        self.decrement_col_value(c, size)
        self.set_value(r, c, 't')
        self.set_value(r - 1, c, 'w')
        self.set_value(r, c - 1, 'w')
        self.set_value(r, c + 1, 'w')
        self.set_value(r - 1, c - 1, 'w')
        self.set_value(r - 1, c + 1, 'w')
        self.decrement_row_value(r, 1)

        if size > 3:
            r += 1
            self.set_value(r, c, 'm')
            self.set_value(r, c - 1, 'w')
            self.set_value(r, c + 1, 'w')
            self.decrement_row_value(r, 1)

        if size > 2:
            r += 1
            self.set_value(r, c, 'm')
            self.set_value(r, c - 1, 'w')
            self.set_value(r, c + 1, 'w')
            self.decrement_row_value(r, 1)

        r += 1
        self.set_value(r, c, 'b')
        self.set_value(r, c - 1, 'w')
        self.set_value(r, c + 1, 'w')
        self.set_value(r + 1, c, 'w')
        self.set_value(r + 1, c - 1, 'w')
        self.set_value(r + 1, c + 1, 'w')
        self.decrement_row_value(r, 1)

    
    def insert_hboat(self, r: int, c: int, size: int):
        self.boats[-size] -= 1
        self.decrement_row_value(r, size)
        self.set_value(r, c, 'l')
        self.set_value(r, c - 1, 'w')
        self.set_value(r - 1, c, 'w')
        self.set_value(r + 1, c, 'w')
        self.set_value(r - 1, c - 1, 'w')
        self.set_value(r + 1, c - 1, 'w')
        self.decrement_col_value(c, 1)

        if size > 3:
            c += 1
            self.set_value(r, c, 'm')
            self.set_value(r - 1, c, 'w')
            self.set_value(r + 1, c, 'w')
            self.decrement_col_value(c, 1)

        if size > 2:
            c += 1
            self.set_value(r, c, 'm')
            self.set_value(r - 1, c, 'w')
            self.set_value(r + 1, c, 'w')
            self.decrement_col_value(c, 1)

        c += 1
        self.set_value(r, c, 'r')
        self.set_value(r, c + 1, 'w')
        self.set_value(r - 1, c, 'w')
        self.set_value(r + 1, c, 'w')
        self.set_value(r - 1, c + 1, 'w')
        self.set_value(r + 1, c + 1, 'w')
        self.decrement_col_value(c, 1)
    
    def add_hint(self, r: int, c: int, value: str):
        if not self.is_pos_valid(r, c):
            return

        self.hints[r][c] = value

        if (value == 'W'):
            self.set_value(r, c, 'w')
            return

        self.set_value(r - 1, c - 1, 'w')
        self.set_value(r - 1, c + 1, 'w')
        self.set_value(r + 1, c - 1, 'w')
        self.set_value(r + 1, c + 1, 'w')

        if (value == 'M'):
            return

        if (value == 'C'):
            # .  .  . (row - 1)
            # .  c  . (row)
            # .  .  . (row + 1)
            self.set_value(r, c, 'c')
            self.set_value(r - 1, c, 'w')
            self.set_value(r + 1, c, 'w')
            self.set_value(r, c - 1, 'w')
            self.set_value(r, c + 1, 'w')
            self.decrement_col_value(c, 1)
            self.decrement_row_value(r, 1)
            self.boats[-1] -= 1
        elif (value == 'R'):
            # . . . (row - 1)
            # ? r . (row)
            # . . . (row + 1)
            self.set_value(r - 1, c, 'w')
            self.set_value(r + 1, c, 'w')
            self.set_value(r, c + 1, 'w')
        elif (value == 'L'):
            # . . . (row - 1)
            # . l ? (row)
            # . . . (row + 1)
            self.set_value(r - 1, c, 'w')
            self.set_value(r + 1, c, 'w')
            self.set_value(r, c - 1, 'w')
        elif (value == 'T'):
            # . . . (row - 1)
            # . t . (row)
            # . ? . (row + 1)
            self.set_value(r - 1, c, 'w')
            self.set_value(r, c - 1, 'w')
            self.set_value(r, c + 1, 'w')
        elif (value == 'B'):
            # . ? . (row - 1)
            # . b . (row)
            # . . . (row + 1)
            self.set_value(r + 1, c, 'w')
            self.set_value(r, c - 1, 'w')
            self.set_value(r, c + 1, 'w')


    def print(self):
        line = ''

        for r in iterator10:
            for c in iterator10:
                value = self.hints[r][c]

                if (value is None):
                    value = self.grid[r][c]

                    if (value == 'w'):
                        value = '.'
                    if (value is None):
                        value = '?'

                line += value
            
            line += '\n'

        print(line[:-1])

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
        board = Board(row_values, col_values)

        for _ in range(int(stdin.readline())):
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
        return self.id < other.id


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(BimaruState(board))

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        possible_actions = []

        size = 4

        for i in iterator4:
            if state.board.boats[i] > 0:
                size -= i
                break
        else:
            return possible_actions
            
        if (size == 1):
            for r in iterator10:
                for c in iterator10:
                    if state.board.fits_cboat(r, c):
                        possible_actions.append((r, c, size))

            return possible_actions
        
        for i in iterator10:
            for j in range(11 - size):
                if state.board.fits_hboat(i, j, size):
                    possible_actions.append((i, j, size, 'H'))

                if state.board.fits_vboat(j, i, size):
                    possible_actions.append((j, i, size, 'V'))

        return possible_actions

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        board = state.board.clone()

        if action[2] == 1:
            board.insert_cboat(action[0], action[1])
        elif action[3] == 'H':
            board.insert_hboat(action[0], action[1], action[2])
        else:
            board.insert_vboat(action[0], action[1], action[2])

        return BimaruState(board)

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        board = state.board

        for r in iterator10:
            for c in iterator10:
                value = board.grid[r][c]

                if (value is None):
                    return False

                hint = board.hints[r][c]

                if (hint is not None) and (value != hint.lower()):
                    return False
                
        if sum(board.row_values) + sum(board.boats):
            return False

        return True

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        return 1

    # TODO: outros metodos da classe


def example1():
    board = Board.parse_instance()

    print(board.adjacent_vertical_values(3, 3))
    print(board.adjacent_horizontal_values(3, 3))

    print(board.adjacent_vertical_values(1, 0))
    print(board.adjacent_horizontal_values(1, 0))


def example2():
    board = Board.parse_instance()

    problem = Bimaru(board)
    initial_state = BimaruState(board)

    print(initial_state.board.get_value(3, 3))

    result_state = problem.result(initial_state, (3, 3, 'w'))

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
