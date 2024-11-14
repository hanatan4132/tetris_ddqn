import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random

style.use("ggplot")

class Tetris:
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.hold_piece = None
        self.can_hold = True
        self.combo = 0
        self.clear = False
        self.max_combo = 0
        self.lines_cleared = 0
        self.reset()

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        self.hold_piece = None
        self.can_hold = True
        self.max_combo = 0
        self.combo = 0
        self.lines_cleared = 0
        self.clear = False
        return self.get_state_properties(self.board)
    
    def rotate(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array
    #---------------------------------------------------------------------------------------
    #  获取当前游戏状态的一些属性
    #---------------------------------------------------------------------------------------
    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        unobstructed_rows = self.check_unobstructed_rows(board)
        consecutive_clearable = self.get_consecutive_clearable_rows(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height, self.combo,self.max_combo, unobstructed_rows, consecutive_clearable])
    
    def get_consecutive_clearable_rows(self, board):
        consecutive = 0
        for row in range(self.height - 1, -1, -1):
            row_data = board[row]
            
            # 检查左右两端是否有像素
            if row_data[0] == 0 and row_data[self.width - 1] == 0:
                break  # 如果两端都没有像素，跳过这一行
    
            # 找到最左边和最右边有像素的索引
            left_index = next((i for i, cell in enumerate(row_data) if cell != 0), None)
            right_index = next((i for i, cell in enumerate(reversed(row_data)) if cell != 0), None)
            right_index = self.width - 1 - right_index  # 修正右侧索引
    
            if left_index is None or right_index is None:
                break  # 如果行中没有任何像素，跳过这一行
    
            # 确保没有空洞：在left_index和right_index之间没有任何空格
            if all(cell != 0 for cell in row_data[left_index:right_index + 1]):
                # 检查整行空格数量是否不超过4个
                empty_count = sum(1 for cell in row_data if cell == 0)
                if empty_count <= 4:
                    consecutive += 1  # 满足条件，增加计数
                else:
                    break  # 如果空格超过4个，停止
            else:
                break  # 如果有空洞，停止
    
        return consecutive

    def check_unobstructed_rows(self, board):
        unobstructed_rows = 0
        for row in range(self.height - 1, -1, -1):  # 從底部開始向上檢查
            row_filled = sum(1 for cell in board[row] if cell != 0)
            if row_filled == self.width:
                unobstructed_rows += 1  # 完全填滿的行也算作無遮擋
            elif row_filled == 0:
                break  # 如果遇到完全空的行,就停止檢查
            else:
                # 檢查這一行的空位是否有被上面的方塊遮擋
                is_unobstructed = True
                for col in range(self.width):
                    if board[row][col] == 0:  # 如果這個位置是空的
                        # 檢查這個位置上面的所有行
                        for above_row in range(row - 1, -1, -1):
                            if board[above_row][col] != 0:
                                is_unobstructed = False
                                break
                        if not is_unobstructed:
                            break
                if is_unobstructed:
                    unobstructed_rows += 1
                else:
                    break  # 如果遇到有遮擋的行,就停止檢查
        
        return unobstructed_rows
    #---------------------------------------------------------------------------------------
    #  面板中空洞数量
    #---------------------------------------------------------------------------------------
    def get_holes(self, board):
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes
    #---------------------------------------------------------------------------------------
    #  计算游戏面板的凹凸度和亮度
    #---------------------------------------------------------------------------------------
    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height
    
    def get_next_states(self):
        if not self.bag:  # If the bag is empty
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag) # Implement this method to add new pieces to the bag
        hold_piece_id = self.bag[-1]
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)

        # Consider hold piece if available
        if self.can_hold:
            if self.hold_piece is None:
                hold_piece_id = self.bag[-1]  # Next piece becomes hold piece
                hold_piece = [row[:] for row in self.pieces[hold_piece_id]]
            else:
                hold_piece = self.hold_piece
            
            hold_piece_id = self.pieces.index(hold_piece)
            if hold_piece_id == 0:  # O piece
                num_rotations = 1
            elif hold_piece_id == 2 or hold_piece_id == 3 or hold_piece_id == 4:
                num_rotations = 2
            else:
                num_rotations = 4

            for i in range(num_rotations):
                valid_xs = self.width - len(hold_piece[0])
                for x in range(valid_xs + 1):
                    piece = [row[:] for row in hold_piece]
                    pos = {"x": x, "y": 0}
                    while not self.check_collision(piece, pos):
                        pos["y"] += 1
                    self.truncate(piece, pos)
                    board = self.store(piece, pos)
                    states[(x, i, 'hold')] = self.get_state_properties(board)
                hold_piece = self.rotate(hold_piece)

        return states

    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True
            
    def check_collision(self, piece, pos):
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False
    def hold(self):
        if self.can_hold :
            if self.hold_piece is None:
                self.hold_piece = self.piece
                self.new_piece()
            else:
                self.hold_piece, self.piece = self.piece, self.hold_piece
                self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
            self.can_hold = False
            
    def truncate(self, piece, pos):
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                self.last_combo = self.max_combo
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board
    def step(self, action, render=True, video=None):
        if isinstance(action, tuple) and len(action) == 3 and action[2] == 'hold':
            self.hold()
            score = 0
        else:
            self.can_hold = True
            x, num_rotations = action
            self.current_pos = {"x": x, "y": 0}
            for _ in range(num_rotations):
                self.piece = self.rotate(self.piece)

            while not self.check_collision(self.piece, self.current_pos):
                self.current_pos["y"] += 1
                if render:
                    self.render(video)

            overflow = self.truncate(self.piece, self.current_pos)
            if overflow:
                self.gameover = True

            self.board = self.store(self.piece, self.current_pos)

            self.lines_cleared , self.board = self.check_cleared_rows(self.board)
            score = 1 + self.lines_cleared * 10
            if self.lines_cleared > 0:
                self.clear = True
                self.combo += 1
                if self.combo >= self.max_combo:
                    self.max_combo = self.combo
                if self.combo > 7:
                    combo = 7
                else :
                    combo = self.combo
                score = int(score * (1 + combo ** 2)) 
            else:
                self.clear = False
                self.combo = 0
                
            self.score += score
            self.tetrominoes += 1
            self.cleared_lines += self.lines_cleared
            if not self.gameover:
                self.new_piece()
            if self.gameover:
                self.combo = self.max_combo
                self.score -= 2

        
        return score, self.gameover

    def render(self, video=None):
        if not self.gameover:
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((self.width * self.block_size, self.height * self.block_size), Image.NEAREST)
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)

        # Render hold piece
        if self.hold_piece is not None:
            hold_piece_img = [self.piece_colors[p] for row in self.hold_piece for p in row]
            hold_piece_img = np.array(hold_piece_img).reshape((len(self.hold_piece), len(self.hold_piece[0]), 3)).astype(np.uint8)
            hold_piece_img = hold_piece_img[..., ::-1]
            hold_piece_img = Image.fromarray(hold_piece_img, "RGB")
            hold_piece_img = hold_piece_img.resize((len(self.hold_piece[0]) * self.block_size, len(self.hold_piece) * self.block_size), Image.NEAREST)
            hold_piece_img = np.array(hold_piece_img)
            
            # 調整 hold 方塊的位置
            hold_piece_height = len(self.hold_piece) * self.block_size
            hold_piece_width = len(self.hold_piece[0]) * self.block_size
            img[11*self.block_size:11*self.block_size+hold_piece_height, 
                self.width*self.block_size+self.block_size:self.width*self.block_size+self.block_size+hold_piece_width, :] = hold_piece_img

        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Hold:", (self.width * self.block_size + int(self.block_size / 2), 10 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        # 添加 Combo 顯示
        cv2.putText(img, "Combo:", (self.width * self.block_size + int(self.block_size / 2), 14 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.combo),
                    (self.width * self.block_size + int(self.block_size / 2), 15 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "last_Combo:", (self.width * self.block_size + int(self.block_size / 2), 18 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=self.text_color)
        cv2.putText(img, str(self.max_combo),
                    (self.width * self.block_size + int(self.block_size / 2), 19 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        if video:
            video.write(img)

        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1)